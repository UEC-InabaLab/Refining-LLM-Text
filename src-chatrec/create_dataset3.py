import os
import json
from tqdm import tqdm
from oss_llm import Llama_Swallow
from inference_deberta import predict_score_with_deberta
import wandb
import socket

def create_preference_data(user_name :str, dialogue: str, summaries: list, rec_info: str, recommendation_sentence: str, score: int):
    predicted_scores = [predict_score_with_deberta(summary, recommendation_sentence, rec_info) for summary in summaries]
    max_index = predicted_scores.index(max(predicted_scores))

    # 最小値のインデックスを取得
    min_index = predicted_scores.index(min(predicted_scores))
    if score == 1:
        data ={}
        data["user_name"] = user_name
        data["dialogue"] = dialogue
        data["chosen_dialogue_summary"] = summaries[max_index]
        data["rejected_dialogue_summary"] = summaries[min_index]
    else:
        data ={}
        data["user_name"] = user_name
        data["dialogue"] = dialogue
        data["chosen_dialogue_summary"] = summaries[min_index]
        data["rejected_dialogue_summary"] = summaries[max_index]
    return data
    
def create_formatted_data(llm, data: dict) -> list:
    dialogue = data["dialogue"]
    places = data["place"]
    candidates_infos = []
    A_Scores = []
    B_Scores = []
    for place in places:
        candidates_infos.append(place["description"])
        A_Scores.append(place["A_Score"])
        B_Scores.append(place["B_Score"])
    datas = []
    for rec_info, score in zip(candidates_infos, A_Scores):
        if score==1:
            dialogue, generated_summaries = llm.generate_summaries(dialogue=dialogue, user_name="A", num_sentence=5)
            recommendation_sentence = llm.generate_rec_sentence_with_no_dialogue(rec_info)
            data = create_preference_data(user_name="A", dialogue=dialogue, summaries=generated_summaries, recommendation_sentence=recommendation_sentence, rec_info=rec_info, score=score)
            datas.append(data)
    for rec_info, score in zip(candidates_infos, B_Scores):
        if score==1:
            dialogue, generated_summaries = llm.generate_summaries(dialogue=dialogue, user_name="B", num_sentence=5)
            recommendation_sentence = llm.generate_rec_sentence_with_no_dialogue(rec_info)
            data = create_preference_data("B", dialogue, generated_summaries, recommendation_sentence, rec_info, score)
            datas.append(data)
    return datas


def create_dataset3():
    llm = Llama_Swallow()
    input_dir = "../data_ChatRec/experiment_chat_and_rec/"
    output_dir = '../data_ChatRec/datasets_3'
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 入力ディレクトリ内のサブディレクトリを取得
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    # TrainとValidで始まるディレクトリのみを処理
    target_dirs = [d for d in subdirs if d.startswith(('Train', 'Valid'))]
    
    for dir_name in tqdm(target_dirs, desc="Processing directories"):
        dir_path = os.path.join(input_dir, dir_name)
        
        # 各サブディレクトリ用の出力ディレクトリを作成
        subdir_output = os.path.join(output_dir, dir_name)
        os.makedirs(subdir_output, exist_ok=True)
        
        # サブディレクトリ内のファイルを処理
        for filename in tqdm(os.listdir(dir_path), desc=f"Processing files in {dir_name}"):
            input_file_path = os.path.join(dir_path, filename)
            output_file_path = os.path.join(subdir_output, filename)
            
            # 出力ファイルが既に存在する場合はスキップ
            if os.path.exists(output_file_path):
                print(f"File {output_file_path} already exists. Skipping.")
                continue
            
            llm_datas = []
            with open(input_file_path, 'r', encoding='utf-8') as file:
                datas = json.load(file)
                llm_datas += create_formatted_data(llm, datas)
            with open(output_file_path, 'w', encoding='utf-8') as file:
                json.dump(llm_datas, file, ensure_ascii=False, indent=4)
                        
                    
                    
if __name__ == "__main__":
    # wandb の初期化（project名やjob名は適宜変更してください）
    wandb.init(project="dataset3_creation_project", job_type="create_dataset3")
    
    try:
        create_dataset3()
        # 正常終了時の alert
        wandb.alert(
            title="Dataset3 Creation Finished",
            text="The dataset creation process has completed successfully.",
        )
    except Exception as e:
        # エラー発生時の alert
        wandb.alert(
            title="Dataset3 Creation Error",
            text=f"{socket.gethostname()}:An error occurred: {str(e)}",
        )
        raise  # エラーを再送出してプロセス終了
    finally:
        wandb.finish()