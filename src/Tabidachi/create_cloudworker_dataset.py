import os
import random
import json
import argparse
from tqdm import tqdm
from inference_deberta import predict_score_with_deberta
import time
from rec_model import Llama_Swallow as RecModel
from summary_model import Llama_Swallow as SummaryModel
from oss_llm import Llama_Swallow

# Set base directory for path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_ID = [102, 114, 119, 203, 209, 305, 312, 316]
# dataset1を元に推薦を行い，推薦データを作成するコード


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
    combined = list(
        zip(
            data["predicted_score"],
            data["score"],
            data["candidate_info_list"],
            data["recommend_sentences"],
        )
    )

    # `predicted_score` を基準に降順で並び替える
    sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)

    # 再び分解して辞書に格納
    (
        data["predicted_score"],
        data["score"],
        data["candidate_info_list"],
        data["recommend_sentences"],
    ) = zip(*sorted_combined)
    return data


def create_recommend_data(data, rec_llm, sum_llm):
    candidates_list = data["candidates"]
    candidates_info_list = [
        candidate["Summary"] + candidate["Feature"] for candidate in candidates_list
    ]
    scores = data["score"]
    predicted_scores = []
    recommend_sentences = []
    dialogue_summary = sum_llm.generate_summary(data["dialogue"])

    for candidate_info in candidates_info_list:
        recommend_sentence = rec_llm.generate_rec_sentence_with_no_dialogue(
            candidate_info, temperature=0.2
        )
        predicted_score = predict_score_with_deberta(
            dialogue_summary, recommend_sentence, candidate_info
        )
        predicted_scores.append(predicted_score)
        recommend_sentences.append(recommend_sentence)

    reccomend_data = {
        "predicted_score": predicted_scores,
        "score": scores,
        "dialogue_summary": dialogue_summary,
        "candidate_info_list": candidates_info_list,
        "recommend_sentences": recommend_sentences,
    }

    return create_formatted_data(reccomend_data)


# 1つのrec_dataの作成
def main(method="baseline1"):
    """
    Create cloudworker evaluation dataset
    
    Args:
        method: "baseline1" or "proposal" - determines which models to use
    """
    
    # Select models based on method
    if method == "baseline1":
        # Use base LLM without DPO training
        rec_llm = Llama_Swallow()
        sum_llm = Llama_Swallow()
        output_dir = os.path.join(BASE_DIR, "../../data/Tabidachi/cloudworker-dataset-baseline1")
        print("Using baseline1 configuration (base LLM without DPO)")
    elif method == "proposal":
        # Use DPO-trained models
        rec_llm = RecModel("dpo-recommendation-results_1")
        sum_llm = SummaryModel("dpo-summary-results_cloudworker")
        output_dir = os.path.join(BASE_DIR, "../../data/Tabidachi/cloudworker-dataset-proposal")
        print("Using proposal configuration (DPO-trained models)")
    else:
        raise ValueError(f"Invalid method: {method}. Choose 'baseline1' or 'proposal'")
    
    input_dir = os.path.join(BASE_DIR, "../../data/Tabidachi/datasets_1/")

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        print(f"filename:{filename}")
        if int(filename[:3]) in TEST_ID:
            if filename[4] == "1" or filename[4] == "2" or filename[4] == "3":
                output_file_path = os.path.join(output_dir, filename)

                # if os.path.exists(output_file_path):
                #     print(f"File {output_file_path} already exists. Skipping.")
                #     continue

                with open(input_dir + filename, "r", encoding="utf-8") as file:
                    datas = json.load(file)
                    for data in datas[-5:-1]:
                        result = create_recommend_data(
                            data, rec_llm=rec_llm, sum_llm=sum_llm
                        )

                        # 全ての情報を含む出力データを作成
                        output_data = {
                            "dialogue_summary": result["dialogue_summary"],
                            "candidates": [],
                        }

                        # 各候補地の情報を追加
                        for i in range(len(result["predicted_score"])):
                            candidate_data = {
                                "candidate_info": result["candidate_info_list"][i],
                                "recommend_sentence": result["recommend_sentences"][i],
                                "predicted_score": result["predicted_score"][i],
                                "true_score": result["score"][i],
                            }
                            output_data["candidates"].append(candidate_data)

                        # ファイルに保存
                        with open(output_file_path, "w", encoding="utf-8") as file:
                            json.dump(output_data, file, ensure_ascii=False, indent=4)

                        print(f"Saved results to {output_file_path}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create cloudworker evaluation dataset for baseline or proposal method"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="baseline1",
        choices=["baseline1", "proposal"],
        help="Method to use: 'baseline1' (base LLM) or 'proposal' (DPO-trained models)"
    )
    
    args = parser.parse_args()
    
    # Run main with specified method
    print(f"Creating cloudworker dataset for method: {args.method}")
    main(method=args.method)
