import os
import json
import glob
import wandb
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from oss_llm import Llama_Swallow

class RecommendationSystem:
    def __init__(self, llm_client):
        """
        観光地レコメンデーションシステムの初期化
        
        Args:
            llm_client: 対話要約やレコメンデーション文生成を行うLLMクライアント
        """
        self.llm = llm_client
        
    def process_data(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        入力データを処理し、スコア予測のための統合データセットを生成する
        
        Args:
            input_data: 対話履歴と場所情報を含む入力データ
            
        Returns:
            指定された形式のデータセット
        """
        # 対話履歴と場所情報を取得
        dialogue = input_data["dialogue"]
        places = input_data["place"]
        
        # ユーザーAとBの対話要約を生成
        user_a_summary = self.llm.generate_summary(dialogue=dialogue, user_name="A")
        user_b_summary = self.llm.generate_summary(dialogue=dialogue, user_name="B")
        
        # 統合データセットを格納するリスト
        dataset = []
        
        # 観光地ごとにレコメンデーション文を一度だけ生成（キャッシュ）
        recommendation_cache = {}
        for place in places:
            place_name = place["name"]
            if place_name not in recommendation_cache:
                recommendation_cache[place_name] = self.llm.generate_rec_sentence_with_no_dialogue(rec_info=place)
        
        # ユーザーAのデータを生成
        for place in places:
            place_name = place["name"]
            dataset_entry = {
                "dialogue_summary": user_a_summary,
                "recommendation_sentence": recommendation_cache[place_name],
                "candidate_information": place["description"],
                "score": place["A_Score"]
            }
            dataset.append(dataset_entry)
        
        # ユーザーBのデータを生成
        for place in places:
            place_name = place["name"]
            dataset_entry = {
                "dialogue_summary": user_b_summary,
                "recommendation_sentence": recommendation_cache[place_name],
                "candidate_information": place["description"],
                "score": place["B_Score"]
            }
            dataset.append(dataset_entry)
        
        return dataset


def process_all_files(base_dir: str, output_dir: str, subsets: List[str]):
    """
    指定されたディレクトリ内のすべてのJSONファイルを処理します
    
    Args:
        base_dir: 入力データのベースディレクトリ
        output_dir: 出力先のベースディレクトリ
        subsets: 処理するサブセット名のリスト（例: ['Train', 'Valid']）
    """
    # Weights & Biasesのセットアップ
    wandb.init(project="data-processing", name="recommendation-dataset-generation")
    
    try:
        # LLMクライアントの初期化
        llm_client = Llama_Swallow()
        
        # レコメンデーションシステムの初期化
        system = RecommendationSystem(llm_client)
        
        # 出力先ディレクトリを作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 全てのJSONファイルのリストを収集
        all_json_files = []
        for subset in subsets:
            for subset_dir in [f"{subset}-except_for_travel", f"{subset}-no_restriction", f"{subset}-travel"]:
                src_dir = os.path.join(base_dir, subset_dir)
                if os.path.exists(src_dir):
                    json_files = glob.glob(os.path.join(src_dir, "*.json"))
                    all_json_files.extend([(json_file, subset, subset_dir) for json_file in json_files])
        
        total_files = len(all_json_files)
        print(f"Total files to process: {total_files}")
        wandb.log({"total_files": total_files})
        
        # 統計情報を追跡
        processed_count = 0
        error_count = 0
        error_files = []
        
        # 進捗状況を表示するプログレスバーを初期化
        progress_bar = tqdm(all_json_files, desc="Processing files", unit="file")
        
        # すべてのファイルを処理
        for json_file, subset, subset_dir in progress_bar:
            filename = os.path.basename(json_file)
            
            # 現在処理中のファイル名をプログレスバーの説明に表示
            progress_bar.set_description(f"Processing {subset_dir}/{filename}")
            
            try:
                # 出力先のサブセットディレクトリを作成（小文字で作成）
                subset_output_dir = os.path.join(output_dir, subset.lower())
                if not os.path.exists(subset_output_dir):
                    os.makedirs(subset_output_dir)
                
                # 入力データの読み込み
                with open(json_file, 'r', encoding='utf-8') as f:
                    input_data = json.load(f)
                
                # データの処理
                processed_data = system.process_data(input_data)
                
                # 出力ファイル名の決定
                output_file = os.path.join(subset_output_dir, filename)
                
                # 同名ファイルが既に存在する場合、一意の名前にする
                if os.path.exists(output_file):
                    base_name, ext = os.path.splitext(filename)
                    category = subset_dir.split('-', 1)[1]  # except_for_travel, no_restriction, travelの部分を抽出
                    output_file = os.path.join(subset_output_dir, f"{base_name}_{category}{ext}")
                
                # 処理結果の保存
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                
                # 処理したデータの統計情報をプログレスバーとwandbに記録
                places_count = len(input_data.get("place", []))
                entries_count = len(processed_data)
                
                processed_count += 1
                
                # wandbに情報をログ
                wandb.log({
                    "processed_files": processed_count,
                    "progress_percentage": (processed_count / total_files) * 100,
                    "places_count": places_count,
                    "entries_count": entries_count,
                    "current_file": f"{subset_dir}/{filename}"
                })
                
                progress_bar.set_postfix({
                    'places': places_count,
                    'entries': entries_count
                })
                
            except Exception as e:
                error_message = f"Error processing {json_file}: {str(e)}"
                print(f"\n{error_message}")
                
                # エラー情報を記録
                error_count += 1
                error_files.append(f"{subset_dir}/{filename}")
                
                # wandbにエラー情報をログ
                wandb.log({
                    "error_count": error_count,
                    "error_files": error_files,
                    "last_error": error_message
                })
                
                # エラーが発生したことを通知
                wandb.alert(
                    title="Processing Error",
                    text=f"Error occurred while processing file {subset_dir}/{filename}: {str(e)}",
                    level=wandb.AlertLevel.ERROR
                )
        
        # 処理完了の表示
        completion_message = f"Processing completed: {processed_count}/{total_files} files processed, {error_count} errors"
        print(f"\n{completion_message}")
        
        # 処理完了を通知
        wandb.alert(
            title="Processing Complete",
            text=completion_message,
            level=wandb.AlertLevel.INFO
        )
        
        # 統計情報を最終的にログ
        wandb.log({
            "completed": True,
            "processed_files": processed_count,
            "error_files_count": error_count,
            "success_rate": (processed_count - error_count) / total_files * 100 if total_files > 0 else 0
        })
        
    except Exception as e:
        # 予期せぬエラーが発生した場合に通知
        error_message = f"Critical error in processing: {str(e)}"
        print(f"\n{error_message}")
        
        wandb.alert(
            title="Critical Error",
            text=error_message,
            level=wandb.AlertLevel.ERROR
        )
        
        # エラー情報をログ
        wandb.log({
            "critical_error": True,
            "error_message": error_message
        })
        
        raise
    
    finally:
        # wandbセッションを終了
        wandb.finish()


def main():
    """
    メイン処理関数
    """
    # 入力データのベースディレクトリ
    base_dir = "../data_ChatRec/experiment_chat_and_rec"
    
    # 出力先のディレクトリ
    output_dir = "../data_ChatRec/dataset2"
    
    # 処理するサブセットのリスト - TestからTrainに変更
    subsets = ["Train", "Valid"]  # Testは除外、Trainを追加
    
    # すべてのファイルを処理
    process_all_files(base_dir, output_dir, subsets)
    
    print(f"Data processing completed. Output directory: {output_dir}")


if __name__ == "__main__":
    main()