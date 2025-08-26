import numpy as np
import os
from tqdm import tqdm
import json


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
            for k in k_values:
                # HitRate@k: 上位kに正解ラベル (1) が含まれるか
                hit_rate = 1 if np.any(actual_scores[:k]) else 0
                hit_rates[k].append(hit_rate)

                # MRR@k: 上位kに正解ラベルが存在する場合、その逆数
                rank = (
                    np.argmax(actual_scores[:k]) + 1 if np.any(actual_scores[:k]) else 0
                )
                mrr = 1 / rank if rank > 0 else 0
                mrrs[k].append(mrr)

    # 平均値を計算
    avg_hit_rates = {k: np.mean(hit_rates[k]) for k in k_values}
    avg_mrrs = {k: np.mean(mrrs[k]) for k in k_values}

    return {"avg_hit_rates@": avg_hit_rates, "avg_mrrs@": avg_mrrs}


def evaluate(input_dir="../data/recommend_data_proposal_10/"):
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(input_dir, exist_ok=True)
    input_datas = []
    for filename in tqdm(os.listdir(input_dir)):
        if filename[4] == "1" or filename[4] == "2" or filename[4] == "3":
            with open(input_dir + filename, "r", encoding="utf-8") as file:
                datas = json.load(file)
                input_datas.append(datas)
    metrics = calculate_metrics(input_datas, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    return metrics


if __name__ == "__main__":
    # metrics = evaluate(input_dir="../data/recommend_data_ablation1_2/")
    # print(metrics)
    dict_list = []
    for i in [0, 1, 2, 3, 4]:
        score = evaluate(f"../data/recommend_data_proposal_{i+1}/")
        dict_list.append(score)
        # 例として dict_list[0] の構造を元にキーを取り出す
    avg_dict = {}
    for metric_key in dict_list[0].keys():
        # metric_key は 'avg_hit_rates@' など
        avg_dict[metric_key] = {}

        # metric_key の下位キー (1,2,...,10) をループ
        for sub_key in dict_list[0][metric_key].keys():
            # 全辞書の同一位置の値を足し合わせる
            total = 0
            for d in dict_list:
                total += d[metric_key][sub_key]

            # 値の平均を計算
            avg_dict[metric_key][sub_key] = total / len(dict_list)

    print(avg_dict)
