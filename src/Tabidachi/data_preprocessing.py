import json
import os
import csv
import shutil
from logging import getLogger

# Set base directory for path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logger = getLogger(__name__)

FILE_PATH = os.path.join(BASE_DIR, "../../data/Tabidachi/annotation_data/annotations/")
OUTPUT_FILE_PATH = os.path.join(BASE_DIR, "../../data/Tabidachi/formatted_data/")
FILE_LIST = os.listdir(FILE_PATH)
SPOT_FILE_PATH = os.path.join(BASE_DIR, "../../data/Tabidachi/annotation_data/spot_info.json")

# spot_info.jsonのSightIDを入力してDetailを取得
def get_details_from_id(id: str):
    with open(SPOT_FILE_PATH, 'r', encoding='utf-8') as file:
        spot_data = json.load(file)
    for spot in spot_data:
        if spot["SightID"] == id:
            return spot["Detail"][0]
        else:
            logger.debug(f"ID {id} not found in spot_info.json")

# annotation_data/のデータからformatted_data/を作成する
# 推薦直前までの対話履歴と観光地候補，推薦観光地をまとめる
def create_formatted_data():
    output_dir = os.path.join(BASE_DIR, '../../data/Tabidachi/formatted_data')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        logger.info(f"{output_dir} は削除されました。")
    
    # 新しいディレクトリを作成
    os.makedirs(output_dir)
    logger.info(f"{output_dir} を再作成しました。")

    annotation_error_cnt = 0
    # 対話データの整形
    for file in FILE_LIST:
        # 整形後のデータを格納するリスト
        with open(FILE_PATH + file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        formatted_data = []

        # 直前までの対話履歴を保持するリスト
        context_history = []
        for entry in data:

            for annotation in entry["annotation"]:
                # sightのdisplayとmentionを取得
                display_ids = annotation["sight"].get("display")
                mentioned_ids = annotation["sight"].get("mention")
                if isinstance(mentioned_ids, list):
                    for mentioned_id in mentioned_ids:
                        # アノテーションエラーを弾く
                        if display_ids is not None and mentioned_id not in display_ids and mentioned_id != "None":
                            logger.info(f"Annotation Error in {file}: {mentioned_id} is not in display_ids")
                            annotation_error_cnt += 1
                        else:
                            if display_ids is not None and mentioned_ids is not None:
                                formatted_entry = {
                                    "context": context_history.copy(),
                                    "candidates": [{"id": id, "detail":get_details_from_id(id)} for id in display_ids],
                                    "mentioned": {"id":mentioned_id,"detail":get_details_from_id(mentioned_id)}
                                    }
                            formatted_data.append(formatted_entry)
            # 対話履歴に追加
            context_history.append({"speaker": entry["speaker"], "utterance": entry["utterance"]})

        # 必要に応じてファイルに保存
        with open(f"{OUTPUT_FILE_PATH}{file}", 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    create_formatted_data()