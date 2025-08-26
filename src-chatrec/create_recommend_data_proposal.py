import os
import json
from tqdm import tqdm
from rec_model import Llama_Swallow as RecModel
from summary_model import Llama_Swallow as SummaryModel
from inference_deberta import predict_score_with_deberta
import time
import wandb
import socket

def create_formatted_data(data, user):
    """
    Sort predictions by predicted score in descending order for a specific user.
    
    Args:
        data: Dictionary containing place data and predicted scores
        user: User identifier ('A' or 'B')
        
    Returns:
        Dictionary with sorted predicted_score and score tuples
    """
    user_key = f"{user}_Score"
    scores = [item[user_key] for item in data["place"]]
    combined = list(zip(data[f"predicted_score_{user}"], scores))

    # Sort by predicted_score in descending order
    sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)

    # Unpack the sorted results
    predicted_scores, scores = zip(*sorted_combined)
    
    return {
        "predicted_score": predicted_scores,
        "score": scores
    }


def create_recommend_data(data, rec_llm, summary_llm):
    """
    Create recommendation data for both users A and B.
    
    Args:
        data: Input data dictionary
        llm: Language model instance
        
    Returns:
        Dictionary with prediction data for users A and B
    """
    places = data["place"]
    candidates_info_list = [place["description"] for place in places]
    
    # 会話の要約をユーザーごとに生成
    dialogue_summary_A = summary_llm.generate_summary(data["dialogue"], "A")
    dialogue_summary_B = summary_llm.generate_summary(data["dialogue"], "B")
    
    predicted_scores_A = []
    predicted_scores_B = []
    
    for candidate_info in candidates_info_list:
        # ユーザーAの予測スコア生成
        recommend_sentence_A = rec_llm.generate_rec_sentence_with_no_dialogue(candidate_info, temperature=0.2)
        predicted_score_A = predict_score_with_deberta(dialogue_summary_A, recommend_sentence_A, candidate_info)
        predicted_scores_A.append(predicted_score_A)
        
        # ユーザーBの予測スコア生成
        recommend_sentence_B = rec_llm.generate_rec_sentence_with_no_dialogue(candidate_info, temperature=0.2)
        predicted_score_B = predict_score_with_deberta(dialogue_summary_B, recommend_sentence_B, candidate_info)
        predicted_scores_B.append(predicted_score_B)
    
    # ユーザーごとの予測スコアを保存
    data_with_predictions = data.copy()
    data_with_predictions["predicted_score_A"] = predicted_scores_A
    data_with_predictions["predicted_score_B"] = predicted_scores_B
    
    return {
        "user_A": create_formatted_data(data_with_predictions, "A"),
        "user_B": create_formatted_data(data_with_predictions, "B")
    }

def main():
    start_time = time.time()
    rec_llm = RecModel(model_name="./dpo-recommendation-results_6")
    sum_llm = SummaryModel(model_name="./dpo-summary-results_6")
    input_dir = "../data_ChatRec/experiment_chat_and_rec/"
    output_dir = "../data_ChatRec/recommend_data_proposal_6/"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all Test directories
    for dir_name in os.listdir(input_dir):
        if not dir_name.startswith("Test"):
            continue
        
        test_dir = os.path.join(input_dir, dir_name)
        if not os.path.isdir(test_dir):
            continue
            
        print(f"Processing test directory: {dir_name}")
        
        # Create corresponding output directory
        test_output_dir = os.path.join(output_dir, dir_name)
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Process all JSON files in the test directory
        for filename in tqdm(os.listdir(test_dir)):
            if not filename.endswith('.json'):
                continue
                
            output_file_path = os.path.join(test_output_dir, filename)
            if os.path.exists(output_file_path):
                print(f"File {output_file_path} already exists. Skipping.")
                continue
                
            input_file_path = os.path.join(test_dir, filename)
            with open(input_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                output_data = create_recommend_data(data, rec_llm=rec_llm, summary_llm=sum_llm)
                
            with open(output_file_path, 'w', encoding='utf-8') as file:
                json.dump(output_data, file, ensure_ascii=False, indent=4)
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

def many_main():
    start_time = time.time()
    for i in range(4):
        rec_llm = RecModel(model_name=f"./dpo-recommendation-results_{i+2}")
        sum_llm = SummaryModel(model_name=f"./dpo-summary-results_{i+2}")
        input_dir = "../data_ChatRec/experiment_chat_and_rec/"
        output_dir = f"../data_ChatRec/recommend_data_proposal_{i+2}/"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all Test directories
        for dir_name in os.listdir(input_dir):
            if not dir_name.startswith("Test"):
                continue
            
            test_dir = os.path.join(input_dir, dir_name)
            if not os.path.isdir(test_dir):
                continue
                
            print(f"Processing test directory: {dir_name}")
            
            # Create corresponding output directory
            test_output_dir = os.path.join(output_dir, dir_name)
            os.makedirs(test_output_dir, exist_ok=True)
            
            # Process all JSON files in the test directory
            for filename in tqdm(os.listdir(test_dir)):
                if not filename.endswith('.json'):
                    continue
                    
                output_file_path = os.path.join(test_output_dir, filename)
                if os.path.exists(output_file_path):
                    print(f"File {output_file_path} already exists. Skipping.")
                    continue
                    
                input_file_path = os.path.join(test_dir, filename)
                with open(input_file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    output_data = create_recommend_data(data, rec_llm=rec_llm, summary_llm=sum_llm)
                    
                with open(output_file_path, 'w', encoding='utf-8') as file:
                    json.dump(output_data, file, ensure_ascii=False, indent=4)
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    wandb.init(project="recommend_data_proposal", job_type="proposal")
    try:
        # many_main()
        main()
        # 正常終了時の alert
        wandb.alert(
            title="recommend_data_proposal Creation Finished",
            text="The dataset creation process has completed successfully.",
        )
    except Exception as e:
        # エラー発生時の alert
        wandb.alert(
            title="recommend_data_proposal Creation Error",
            text=f"{socket.gethostname()}:An error occurred: {str(e)}",
        )
        raise  # エラーを再送出してプロセス終了
    finally:
        wandb.finish()