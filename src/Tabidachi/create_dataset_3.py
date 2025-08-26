import os
import json
from tqdm import tqdm
from oss_llm import Llama_Swallow
from inference_deberta import predict_score_with_deberta
import wandb
import socket

# Set base directory for path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_ID = [
    101,
    103,
    104,
    105,
    106,
    107,
    109,
    110,
    111,
    112,
    113,
    115,
    116,
    117,
    118,
    121,
    122,
    123,
    124,
    125,
    201,
    202,
    204,
    205,
    207,
    208,
    210,
    302,
    303,
    304,
    306,
    307,
    308,
    309,
    310,
    311,
    313,
    314,
    315,
    318,
    319,
    320,
]
TEST_ID = [102, 114, 119, 203, 209, 305, 312, 316]
VALID_ID = [108, 120, 206, 301, 317]

# 3つのGPUでデータセット3の作成
TRAIN_VALID_ID_CLOVA = [
    122,
    123,
    124,
    125,
    201,
    202,
    204,
    205,
    207,
    208,
    210,
    302,
    303,
    304,
    306,
    307,
    101,
    103,
    104,
    105,
    106,
]
TRAIN_VALID_ID_BARD = [
    308,
    309,
    310,
    311,
    313,
    314,
    315,
    318,
    319,
    320,
    108,
    120,
    206,
    301,
    317,
    107,
    109,
    110,
    111,
    112,
    113,
    115,
    116,
    117,
    118,
    121,
]
print(len(TRAIN_VALID_ID_CLOVA + TRAIN_VALID_ID_BARD), flush=True)
# TRAIN_VALID_ID = TRAIN_ID + VALID_ID


def create_preference_data(
    all_short_dialogue_summary: str,
    summaries: list,
    rec_info: str,
    recommendation_sentence: str,
    score: int,
):
    predicted_scores = [
        predict_score_with_deberta(summary, recommendation_sentence, rec_info)
        for summary in summaries
    ]
    max_index = predicted_scores.index(max(predicted_scores))

    # 最小値のインデックスを取得
    min_index = predicted_scores.index(min(predicted_scores))
    if score == 1:
        data = {}
        data["all_short_dialogue_summary"] = all_short_dialogue_summary
        data["chosen_dialogue_summary"] = summaries[max_index]
        data["rejected_dialogue_summary"] = summaries[min_index]
    else:
        data = {}
        data["all_short_dialogue_summary"] = all_short_dialogue_summary
        data["chosen_dialogue_summary"] = summaries[min_index]
        data["rejected_dialogue_summary"] = summaries[max_index]
    return data


def create_formatted_data(llm, data: dict) -> list:
    dialogue = data["dialogue"]
    candidates = data["candidates"]
    candidates_infos = [
        candidate["Summary"] + candidate["Feature"] for candidate in candidates
    ]  # candidateのSummaryとFeatureを観光地情報として使用
    # print(f"candidates_infos:{candidates_infos}")
    scores = data["score"]
    datas = []
    for rec_info, score in zip(candidates_infos, scores):
        if score == 1:
            all_short_dialogue_summary, generated_summaries = llm.generate_summaries(
                dialogue=dialogue, num_sentence=5
            )
            recommendation_sentence = llm.generate_rec_sentence_with_no_dialogue(
                rec_info
            )
            data = create_preference_data(
                all_short_dialogue_summary=all_short_dialogue_summary,
                summaries=generated_summaries,
                recommendation_sentence=recommendation_sentence,
                rec_info=rec_info,
                score=score,
            )
            datas.append(data)
    return datas


def create_datasets_3():
    llm = Llama_Swallow()
    input_dir = os.path.join(BASE_DIR, "../../data/Tabidachi/datasets_1/")
    output_dir = os.path.join(BASE_DIR, "../../data/Tabidachi/datasets_3/")
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    for filename in tqdm(os.listdir(input_dir)):
        # if int(filename[:3]) in TRAIN_ID or int(filename[:3]) in VALID_ID:  #本来のもの
        if int(filename[:3]) in TRAIN_VALID_ID_CLOVA:  # 3つのGPUでデータセット3を作成
            if filename[4] == "1" or filename[4] == "2" or filename[4] == "3":
                output_file_path = os.path.join(output_dir, filename)
                if os.path.exists(output_file_path):
                    print(f"File {output_file_path} already exists. Skipping.")
                    continue
                llm_datas = []
                with open(input_dir + filename, "r", encoding="utf-8") as file:
                    datas = json.load(file)
                    for data in datas:
                        llm_datas += create_formatted_data(llm, data)
                        # print(llm_datas)
                    with open(output_file_path, "w", encoding="utf-8") as file:
                        json.dump(llm_datas, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # wandb の初期化（project名やjob名は適宜変更してください）
    wandb.init(project="datasets_3_creation_project", job_type="create_datasets_3")

    try:
        create_datasets_3()
        # 正常終了時の alert
        wandb.alert(
            title="datasets_3 Creation Finished",
            text="The dataset creation process has completed successfully.",
        )
    except Exception as e:
        # エラー発生時の alert
        wandb.alert(
            title="datasets_3 Creation Error",
            text=f"{socket.gethostname()}:An error occurred: {str(e)}",
        )
        raise  # エラーを再送出してプロセス終了
    finally:
        wandb.finish()
