import os
import random
import json
from tqdm import tqdm
from inference_deberta import predict_score_with_deberta
import time
from rec_model import Llama_Swallow as RecModel
from summary_model import Llama_Swallow as SummaryModel

# Set base directory for path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_ID = [102, 114, 119, 203, 209, 305, 312, 316]
# 　dataset1を元に推薦を行い，推薦データを作成するコード


def create_formatted_data(data):
    """_summary_

    Args:
        data = {
    "predicted_score": [0.1, 0.4, 0.45, 0.5, 0.2],
    "score": [0, 0, 1, 0, 0]
    }


    Returns:
        {
            "predicted_score": (0.5, 0.45, 0.4, 0.2, 0.1),
            "score": (0, 1, 0, 0, 0)
        }
    """
    # `zip` を使って `predicted_score` と `score` を結びつける
    combined = list(zip(data["predicted_score"], data["score"]))

    # `predicted_score` を基準に降順で並び替える
    sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)

    # 再び分解して辞書に格納
    data["predicted_score"], data["score"] = zip(*sorted_combined)
    return data


def create_recommend_data(data, rec_llm, sum_llm):
    candidates_list = data["candidates"]
    candidates_info_list = [
        candidate["Summary"] + candidate["Feature"] for candidate in candidates_list
    ]
    scores = data["score"]
    predicted_scores = []
    dialogue_summary = sum_llm.generate_summary(data["dialogue"])
    for candidate_info in candidates_info_list:
        recommend_sentence = rec_llm.generate_rec_sentence_with_no_dialogue(
            candidate_info, temperature=0.2
        )
        predicted_score = predict_score_with_deberta(
            dialogue_summary, recommend_sentence, candidate_info
        )
        predicted_scores.append(predicted_score)
    reccomend_data = {"predicted_score": predicted_scores, "score": scores}
    return create_formatted_data(reccomend_data)


# 1つのrec_dataの作成
def main():
    start_time = time.time()
    rec_llm = RecModel("dpo-recommendation-results_1")
    sum_llm = SummaryModel("dpo-summary-results_1")
    input_dir = os.path.join(BASE_DIR, "../../data/Tabidachi/datasets_1/")
    # output_dir = os.path.join(BASE_DIR, "../../data/Tabidachi/recommend_data_proposal/")
    output_dir = os.path.join(BASE_DIR, "../../data/Tabidachi/recommend_data_proposal_1/")
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    for filename in tqdm(os.listdir(input_dir)):
        print(f"filename:{filename}")
        if int(filename[:3]) in TEST_ID:
            if filename[4] == "1" or filename[4] == "2" or filename[4] == "3":
                output_file_path = os.path.join(output_dir, filename)
                if os.path.exists(output_file_path):
                    print(f"File {output_file_path} already exists. Skipping.")
                    continue
                output_datas = []
                with open(input_dir + filename, "r", encoding="utf-8") as file:
                    datas = json.load(file)
                    for data in datas:
                        output_datas.append(
                            create_recommend_data(
                                data, rec_llm=rec_llm, sum_llm=sum_llm
                            )
                        )
                    with open(output_file_path, "w", encoding="utf-8") as file:
                        json.dump(output_datas, file, ensure_ascii=False, indent=4)
        # 実行終了時間を記録
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


# 複数のrec_dataの作成(ベストハイパラで複数のモデルを学習したものを使用)
def many_main():
    start_time = time.time()
    for i in range(5):
        rec_llm = RecModel(model_name=f"dpo-recommendation-results_{i+1}")
        sum_llm = SummaryModel(model_name=f"dpo-summary-results_{i+1}")
        input_dir = os.path.join(BASE_DIR, "../../data/Tabidachi/datasets_1/")
        output_dir = os.path.join(BASE_DIR, f"../../data/Tabidachi/recommend_data_proposal_{i+1}/")
        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(output_dir, exist_ok=True)
        for filename in tqdm(os.listdir(input_dir)):
            print(f"filename:{filename}")
            if int(filename[:3]) in TEST_ID:
                if filename[4] == "1" or filename[4] == "2" or filename[4] == "3":
                    output_file_path = os.path.join(output_dir, filename)
                    if os.path.exists(output_file_path):
                        print(f"File {output_file_path} already exists. Skipping.")
                        continue
                    output_datas = []
                    with open(input_dir + filename, "r", encoding="utf-8") as file:
                        datas = json.load(file)
                        for data in datas:
                            output_datas.append(
                                create_recommend_data(
                                    data, rec_llm=rec_llm, sum_llm=sum_llm
                                )
                            )
                        with open(output_file_path, "w", encoding="utf-8") as file:
                            json.dump(output_datas, file, ensure_ascii=False, indent=4)
        # 実行終了時間を記録
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    many_main()
