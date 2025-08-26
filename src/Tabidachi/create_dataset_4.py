import os
import json
from tqdm import tqdm
from oss_llm import Llama_Swallow
from inference_deberta import predict_score_with_deberta
import wandb

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

TRAIN_VALID_ID = TRAIN_ID + VALID_ID


def create_preference_data(
    dialogue_summary: str,
    rec_sentence_list: list,
    candidate_information: str,
    score: int,
):
    predicted_scores = [
        predict_score_with_deberta(
            dialogue_summary, recommendation_sentence, candidate_information
        )
        for recommendation_sentence in rec_sentence_list
    ]
    max_index = predicted_scores.index(max(predicted_scores))

    # 最小値のインデックスを取得
    min_index = predicted_scores.index(min(predicted_scores))
    if score == 1:
        data = {}
        data["candidate_information"] = candidate_information
        data["chosen_recommendation_sentence"] = rec_sentence_list[max_index]
        data["rejected_recommendation_sentence"] = rec_sentence_list[min_index]
    else:
        data = {}
        data["candidate_information"] = candidate_information
        data["chosen_recommendation_sentence"] = rec_sentence_list[min_index]
        data["rejected_recommendation_sentence"] = rec_sentence_list[max_index]
    return data


def create_formatted_data(llm, data: dict) -> list:
    score = data["score"]
    if score == 1:
        dialogue_summary = data["dialogue_summary"]
        candidate_information = data["candidate_information"]
        recommendation_sentence_list = [data["recommendation_sentence"]]
        for i in range(
            9
        ):  # 元々(datasets2で)一つは推薦文を生成しているため，ここでは9つの推薦文を生成
            recommendation_sentence_list.append(
                llm.generate_rec_sentence_with_no_dialogue(
                    candidate_information, temperature=1.0
                )
            )
        data = create_preference_data(
            dialogue_summary, recommendation_sentence_list, candidate_information, score
        )
        return data
    return None


def create_datasets_4():
    llm = Llama_Swallow()
    input_dir = os.path.join(BASE_DIR, "../../data/Tabidachi/datasets_2/")
    output_dir = os.path.join(BASE_DIR, "../../data/Tabidachi/datasets_4/")
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    for filename in tqdm(os.listdir(input_dir)):
        if int(filename[:3]) in TRAIN_VALID_ID:
            if filename[4] == "1" or filename[4] == "2" or filename[4] == "3":
                output_file_path = os.path.join(output_dir, filename)
                if os.path.exists(output_file_path):
                    print(f"File {output_file_path} already exists. Skipping.")
                    continue
                llm_datas = []
                with open(input_dir + filename, "r", encoding="utf-8") as file:
                    datas = json.load(file)
                    for data in datas:
                        preference_data = create_formatted_data(llm, data)
                        if preference_data != None:
                            llm_datas.append(preference_data)
                        # print(llm_datas)
                    with open(output_file_path, "w", encoding="utf-8") as file:
                        json.dump(llm_datas, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # wandb の初期化（project名やjob名は適宜変更してください）
    wandb.init(
        project="datasets_4_creation_project",
        job_type="create_datasets_4",
        dir="/tmp/wandb",
    )  # dirを指定

    try:
        create_datasets_4()
        # 正常終了時の alert
        wandb.alert(
            title="datasets_4 Creation Finished",
            text="The dataset creation process has completed successfully.",
        )
    except Exception as e:
        # エラー発生時の alert
        wandb.alert(
            title="datasets_4 Creation Error",
            text=f"An error occurred: {str(e)}",
        )
        raise  # エラーを再送出してプロセス終了
    finally:
        wandb.finish()
