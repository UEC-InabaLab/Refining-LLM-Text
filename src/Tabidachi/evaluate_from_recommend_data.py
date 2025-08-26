import numpy as np
import os
from tqdm import tqdm
import json
import argparse

# Set base directory for path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def calculate_metrics(data, k_values):
    """
    HitRate@k と MRR@k を計算する
    :param data: 入力データ (複数のセットで構成)
    :param k_values: 計算する k のリスト (例: [1, 2, 3, 4, 5])
    :return: 各 k に対する HitRate と MRR のリスト
    """
    hit_rates = {k: [] for k in k_values}
    mrrs = {k: [] for k in k_values}

    for data_group in data:  # data は複数のセットを含む
        for entry in data_group:
            predicted_score = np.array(entry["predicted_score"])
            actual_scores = np.array(entry["score"])
            
            # predicted_score でソートするためのインデックスを取得
            sorted_indices = np.argsort(predicted_score)[::-1]  # 降順にソート
            sorted_actual_scores = actual_scores[sorted_indices]
            
            for k in k_values:
                # HitRate@k: 上位kに正解ラベル (1) が含まれるか
                hit_rate = 1 if np.any(sorted_actual_scores[:k] == 1) else 0
                hit_rates[k].append(hit_rate)

                # MRR@k: 上位k内の最初の正解ラベルの位置の逆数
                correct_indices = np.where(sorted_actual_scores[:k] == 1)[0]
                if len(correct_indices) > 0:
                    first_correct_idx = correct_indices[0]
                    mrr = 1.0 / (first_correct_idx + 1)
                else:
                    mrr = 0
                mrrs[k].append(mrr)

    # 平均値を計算
    avg_hit_rates = {k: np.mean(hit_rates[k]) for k in k_values}
    avg_mrrs = {k: np.mean(mrrs[k]) for k in k_values}

    return {"avg_hit_rates@": avg_hit_rates, "avg_mrrs@": avg_mrrs}


def evaluate_single_dataset(input_dir, method_name):
    """
    単一のデータセットを評価する
    :param input_dir: 入力ディレクトリパス
    :param method_name: メソッド名（表示用）
    :return: 評価メトリクス
    """
    if not os.path.exists(input_dir):
        print(f"ディレクトリ {input_dir} が見つかりません。")
        return None
    
    input_datas = []
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    print(f"\n{method_name} の評価を開始: {len(json_files)} 個のファイルを処理")
    
    for filename in tqdm(json_files, desc=f"{method_name} のファイルを読み込み中"):
        if filename[4] == "1" or filename[4] == "2" or filename[4] == "3":
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                datas = json.load(file)
                input_datas.append(datas)
    
    if input_datas:
        metrics = calculate_metrics(input_datas, list(range(1, 11)))
        return metrics
    else:
        print(f"{method_name} のデータが読み込めませんでした。")
        return None


def evaluate_multiple_models(base_method, num_models=5):
    """
    複数モデル（1-5）のデータセットを評価して平均を計算する
    :param base_method: ベースメソッド名（proposal, ablation1, ablation2）
    :param num_models: モデル数（デフォルト5）
    :return: 全モデルの平均メトリクス
    """
    k_values = list(range(1, 11))
    all_hit_rates = {k: [] for k in k_values}
    all_mrrs = {k: [] for k in k_values}
    valid_count = 0
    
    for i in range(1, num_models + 1):
        input_dir = os.path.join(BASE_DIR, f"../../data/Tabidachi/recommend_data_{base_method}_{i}/")
        
        if not os.path.exists(input_dir):
            print(f"ディレクトリ {input_dir} が見つかりません。スキップします。")
            continue
        
        metrics = evaluate_single_dataset(input_dir, f"{base_method}_{i}")
        
        if metrics:
            valid_count += 1
            for k in k_values:
                all_hit_rates[k].append(metrics["avg_hit_rates@"][k])
                all_mrrs[k].append(metrics["avg_mrrs@"][k])
            
            print_metrics(metrics, f"{base_method}_{i} の結果")
    
    if valid_count > 0:
        avg_hit_rates = {k: np.mean(all_hit_rates[k]) for k in k_values}
        avg_mrrs = {k: np.mean(all_mrrs[k]) for k in k_values}
        
        return {
            "avg_hit_rates@": avg_hit_rates,
            "avg_mrrs@": avg_mrrs,
            "valid_model_count": valid_count
        }
    else:
        return None


def print_metrics(metrics, title="評価結果"):
    """
    メトリクスを表示する
    :param metrics: 評価メトリクス
    :param title: 表示タイトル
    """
    if not metrics:
        return
    
    print(f"\n=== {title} ===")
    print("\nHit Rate:")
    for k in sorted(metrics["avg_hit_rates@"].keys()):
        print(f"  Hit Rate@{k}: {metrics['avg_hit_rates@'][k]:.4f}")
    
    print("\nMRR:")
    for k in sorted(metrics["avg_mrrs@"].keys()):
        print(f"  MRR@{k}: {metrics['avg_mrrs@'][k]:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Tabidachi推薦データの評価を行う"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["proposal", "baseline1", "baseline2", "ablation1", "ablation2"],
        help="評価するメソッド: proposal, baseline1, baseline2, ablation1, ablation2"
    )
    
    args = parser.parse_args()
    method = args.method
    
    print(f"\n{'='*60}")
    print(f"Tabidachi {method} の評価を開始")
    print(f"{'='*60}")
    
    # proposal と ablation は複数モデル（1-5）を評価
    if method in ["proposal", "ablation1", "ablation2"]:
        print(f"\n{method} の複数モデル（1-5）を自動的に評価します。")
        results = evaluate_multiple_models(method, num_models=5)
        
        if results:
            model_count = results.pop("valid_model_count", 0)
            print(f"\n{'='*60}")
            print(f"最終結果: {model_count}個のモデルの平均（{method}）")
            print(f"{'='*60}")
            print_metrics(results, f"{method} 全モデルの平均")
        else:
            print(f"\n{method} のデータが見つかりませんでした。")
    
    # baseline は単一データセットを評価
    else:
        input_dir = os.path.join(BASE_DIR, f"../../data/Tabidachi/recommend_data_{method}/")
        results = evaluate_single_dataset(input_dir, method)
        
        if results:
            print(f"\n{'='*60}")
            print(f"最終結果: {method}")
            print(f"{'='*60}")
            print_metrics(results, f"{method} の評価結果")
        else:
            print(f"\n{method} のデータが見つかりませんでした。")
    
    print(f"\n評価が完了しました。\n")


if __name__ == "__main__":
    main()
