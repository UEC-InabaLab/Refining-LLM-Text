import numpy as np
import os
import json
from tqdm import tqdm
import argparse

# Set base directory for path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def calculate_metrics(data, k_values):
    """
    HitRate@k と MRR@k を計算する関数
    
    :param data: 各ユーザーのデータを含む辞書のリスト
    :param k_values: 計算する k のリスト (例: [1, 2, 3, ..., 10])
    :return: 各 k に対する HitRate と MRR の辞書
    """
    hit_rates = {k: [] for k in k_values}
    mrrs = {k: [] for k in k_values}
    
    # 各データセットに対して処理
    for entry in data:
        for user_id, user_data in entry.items():
            # スコアを numpy 配列に変換
            predicted_scores = np.array(user_data["predicted_score"])
            actual_scores = np.array(user_data["score"])
            
            # 予測スコアでソートするためのインデックスを取得
            sorted_indices = np.argsort(predicted_scores)[::-1]  # 降順にソート
            
            # ソートされた actual_scores を取得
            sorted_actual_scores = actual_scores[sorted_indices]
            
            for k in k_values:
                # HitRate@k: 上位 k に正解ラベル (1) が含まれるか
                hit_rate = 1 if np.any(sorted_actual_scores[:k] == 1) else 0
                hit_rates[k].append(hit_rate)
                
                # MRR@k: 上位 k 内の最初の正解ラベルの位置の逆数
                # 正解がない場合は 0
                correct_indices = np.where(sorted_actual_scores[:k] == 1)[0]
                if len(correct_indices) > 0:
                    first_correct_idx = correct_indices[0]
                    mrr = 1.0 / (first_correct_idx + 1)  # インデックスは0始まりなので+1
                else:
                    mrr = 0
                mrrs[k].append(mrr)
    
    # 平均値を計算
    avg_hit_rates = {k: np.mean(hit_rates[k]) for k in k_values}
    avg_mrrs = {k: np.mean(mrrs[k]) for k in k_values}
    
    return {"avg_hit_rates@": avg_hit_rates, "avg_mrrs@": avg_mrrs}

def find_all_json_files(directory):
    """
    指定ディレクトリとその全てのサブディレクトリからJSONファイルを再帰的に検索
    
    :param directory: 検索を開始するディレクトリパス
    :return: 全JSONファイルのパスのリスト
    """
    json_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    return json_files

def evaluate_all_json_files(base_directory, method_name=""):
    """
    指定されたベースディレクトリ内の全てのJSONファイルを評価
    
    :param base_directory: ベースディレクトリパス
    :param method_name: メソッド名（表示用）
    :return: 計算されたメトリクス
    """
    k_values = list(range(1, 11))  # @1から@10まで
    all_data = []
    
    if not os.path.exists(base_directory):
        print(f"ディレクトリ {base_directory} が見つかりません。")
        return None
    
    # 全てのJSONファイルを検索
    json_files = find_all_json_files(base_directory)
    print(f"{method_name}: ディレクトリから {len(json_files)} 個のJSONファイルが見つかりました")
    
    # 全てのファイルを読み込む
    desc = f"{method_name} のJSONファイルを読み込み中" if method_name else "JSONファイルを読み込み中"
    for file_path in tqdm(json_files, desc=desc):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                all_data.append(data)
        except Exception as e:
            print(f"ファイル読み込みエラー {file_path}: {e}")
    
    # メトリクスを計算
    if all_data:
        metrics = calculate_metrics(all_data, k_values)
        return metrics
    else:
        print(f"{method_name}: データが読み込めませんでした。")
        return None

def evaluate_multiple_directories(method, num_models=5):
    """
    複数のディレクトリの結果を評価し、平均を計算する（proposalの場合）
    
    :param method: 評価メソッド名
    :param num_models: モデル数（デフォルト5）
    :return: 全ディレクトリの平均メトリクス
    """
    k_values = list(range(1, 11))  # @1から@10まで
    
    # 各kに対する結果を保存するための辞書を初期化
    all_hit_rates = {k: [] for k in k_values}
    all_mrrs = {k: [] for k in k_values}
    
    base_path = os.path.join(BASE_DIR, "../../data/ChatRec")
    directories = [f"recommend_data_{method}_{i}" for i in range(1, num_models + 1)]
    valid_dir_count = 0
    
    # 各ディレクトリを評価
    for dir_name in directories:
        full_path = os.path.join(base_path, dir_name)
        print(f"\n=== {dir_name} の評価 ===")
        
        # ディレクトリが存在するか確認
        if not os.path.exists(full_path):
            print(f"ディレクトリ {full_path} が見つかりません。スキップします。")
            continue
            
        metrics = evaluate_all_json_files(full_path, dir_name)
        
        if metrics:
            valid_dir_count += 1
            # 各kに対する結果を追加
            for k in k_values:
                all_hit_rates[k].append(metrics["avg_hit_rates@"][k])
                all_mrrs[k].append(metrics["avg_mrrs@"][k])
            
            # 個別ディレクトリの結果を表示
            print_results(metrics, f"{dir_name} の結果")
    
    # 全ディレクトリの平均を計算
    if valid_dir_count > 0:
        avg_hit_rates = {k: np.mean(all_hit_rates[k]) for k in k_values}
        avg_mrrs = {k: np.mean(all_mrrs[k]) for k in k_values}
        
        # 最終結果を返す
        return {
            "avg_hit_rates@": avg_hit_rates, 
            "avg_mrrs@": avg_mrrs,
            "valid_dir_count": valid_dir_count
        }
    else:
        print("有効なデータが見つかりませんでした。")
        return None

def print_results(metrics, title="評価結果"):
    """
    結果を出力する
    
    :param metrics: 計算されたメトリクス
    :param title: 出力のタイトル
    """
    if not metrics:
        return
    
    print(f"\n=== {title} ===")
    
    # Hit Rate
    print("\nHit Rate:")
    for k in sorted(metrics["avg_hit_rates@"].keys()):
        print(f"  Hit Rate@{k}: {metrics['avg_hit_rates@'][k]:.4f}")
    
    # MRR
    print("\nMRR:")
    for k in sorted(metrics["avg_mrrs@"].keys()):
        print(f"  MRR@{k}: {metrics['avg_mrrs@'][k]:.4f}")

def main():
    parser = argparse.ArgumentParser(
        description="ChatRec推薦データの評価を行う"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["proposal", "baseline1", "baseline2"],
        help="評価するメソッド: proposal, baseline1, baseline2"
    )
    
    args = parser.parse_args()
    method = args.method
    
    print(f"\n{'='*60}")
    print(f"ChatRec {method} の評価を開始")
    print(f"{'='*60}")
    
    # proposalは複数モデル（1-5）を評価
    if method == "proposal":
        print(f"\n{method} の複数モデル（1-5）を自動的に評価します。")
        avg_metrics = evaluate_multiple_directories(method, num_models=5)
        
        # 最終結果を出力（全ディレクトリの平均）
        if avg_metrics:
            dir_count = avg_metrics.pop("valid_dir_count", 0)
            print(f"\n{'='*60}")
            print(f"最終結果: {dir_count}個のモデルの平均（{method}）")
            print(f"{'='*60}")
            print_results(avg_metrics, f"{method} 全モデルの平均")
        else:
            print(f"\n{method} のデータが見つかりませんでした。")
    
    # baselineは単一データセットを評価
    else:
        base_directory = os.path.join(BASE_DIR, f"../../data/ChatRec/recommend_data_{method}/")
        print(f"\n{method} のデータセットを評価します。")
        
        metrics = evaluate_all_json_files(base_directory, method)
        
        if metrics:
            print(f"\n{'='*60}")
            print(f"最終結果: {method}")
            print(f"{'='*60}")
            print_results(metrics, f"{method} の評価結果")
        else:
            print(f"\n{method} のデータが見つかりませんでした。")
    
    print(f"\n評価が完了しました。\n")


if __name__ == '__main__':
    main()