import os
import json
from tqdm import tqdm

TRAIN_ID = [101, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 115, 116, 117, 118, 121, 122, 123, 124, 125, 201, 202, 204, 205, 207, 208, 210, 302, 303, 304, 306, 307, 308, 309, 310, 311, 313, 314, 315, 318, 319, 320]
TEST_ID = [102, 114, 119, 203, 209, 305, 312, 316]
VALID_ID = [108, 120, 206, 301, 317]

# データセットの特徴量をカウント （それぞれのデータごとに）
def count_places_and_rec_dialogue(input_dir="../data/datasets_1/"):
    dialogue_count = 0 
    places_set = set()
    for filename in tqdm(os.listdir(input_dir)):
            # if int(filename[:3]) in TRAIN_ID or int(filename[:3]) in VALID_ID:  #本来のもの
            if int(filename[:3]) in TEST_ID:  #3つのGPUでデータセット3を作成
                if filename[4]=="1" or filename[4]=="2" or filename[4]=="3":
                    with open(input_dir + filename, 'r', encoding='utf-8') as file:
                        datas = json.load(file)
                        dialogue_count += len(datas)
                        for data in datas:
                            places_set.update(entry["Area"] for entry in data["candidates"])
                        
    print(f"rec_count:{dialogue_count}")
    print("places_count:", len(places_set))


# 発話数をカウント
def count_dialog_switches():
    input_dir = "../data/annotation_data/annotations/"
    switches = 0
    for filename in tqdm(os.listdir(input_dir)):
            # if int(filename[:3]) in TRAIN_ID or int(filename[:3]) in VALID_ID:  #本来のもの
            if int(filename[:3]) in TRAIN_ID:  #3つのGPUでデータセット3を作成
                if filename[4]=="1" or filename[4]=="2" or filename[4]=="3":
                    with open(input_dir + filename, 'r', encoding='utf-8') as file:
                        conversation = json.load(file)
                        if not conversation:
                    # 会話データ自体が空であれば 0
                            return 0
    
    # 最初の発話者を記録
                        previous_speaker = conversation[0]["speaker"]
    
    # 2つ目以降の要素をループして、operator <-> customer のみを切り替わりと数える
                        for data in conversation:
                            current_speaker = data["speaker"]
                            if (previous_speaker != current_speaker) and (
                                (previous_speaker == "operator" and current_speaker == "customer") or
                                (previous_speaker == "customer" and current_speaker == "operator")
                            ):
                                switches += 1
                            previous_speaker = current_speaker
                        
    return switches

if __name__ == "__main__":
    print("switches:", count_dialog_switches())