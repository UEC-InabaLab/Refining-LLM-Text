import os

# Set base directory for path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import os
import json
from tqdm import tqdm
from oss_llm import Llama_Swallow

# データセット1からデータセット2の作成
# 使用するデータセットは対話IDが1,2,3のもの（画面共有なしの対話）


def create_formatted_data(llm, data: dict) -> list:
    dialogue = data["dialogue"]
    candidates = data["candidates"]
    candidates_infos = [
        candidate["Summary"] + candidate["Feature"] for candidate in candidates
    ]  # candidateのSummaryとFeatureを観光地情報として使用
    # print(f"candidates_infos:{candidates_infos}")
    scores = data["score"]
    dialogue_summary = llm.generate_summary(dialogue=dialogue)
    datas = []
    for rec_info, score in zip(candidates_infos, scores):
        data = {}
        rec_sentence = llm.generate_rec_sentence_with_no_dialogue(rec_info=rec_info)
        data["dialogue_summary"] = dialogue_summary
        data["candidate_information"] = rec_info
        data["recommendation_sentence"] = rec_sentence
        data["score"] = score
        datas.append(data)
    # print(datas)
    return datas


def create_datasets_2():
    llm = Llama_Swallow()
    input_dir = os.path.join(BASE_DIR, "../../data/Tabidachi/datasets_1/")
    output_dir = os.path.join(BASE_DIR, "../../data/Tabidachi/datasets_2/")
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    for filename in tqdm(os.listdir(input_dir)):
        if filename[4] == "1" or filename[4] == "2" or filename[4] == "3":
            output_file_path = os.path.join(output_dir, filename)
            llm_datas = []
            with open(input_dir + filename, "r", encoding="utf-8") as file:
                datas = json.load(file)
                for data in datas:
                    llm_datas += create_formatted_data(llm, data)
                    # print(llm_datas)
                with open(output_file_path, "w", encoding="utf-8") as file:
                    json.dump(llm_datas, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    create_datasets_2()
