import os
import json
from tqdm import tqdm
# formatted_dataをもとにdataset_1を作成

# 変換処理を定義
def transform_data(data):
    # 同じ話者の発言をまとめる
    dialogue_parts = []
    current_speaker = None
    current_utterance = ""

    for entry in data["context"]:
        speaker = entry["speaker"]
        utterance = entry["utterance"]
        if speaker == "operator":
            speaker = "A"
        if speaker == "customer":
            speaker = "B"
        if speaker == current_speaker:
            current_utterance += " " + utterance
        else:
            if current_utterance:
                dialogue_parts.append(f"{current_speaker.capitalize()}: {current_utterance}")
            current_speaker = speaker
            current_utterance = utterance

    # 最後の発言を追加
    if current_utterance:
        dialogue_parts.append(f"{current_speaker.capitalize()}: {current_utterance}")

    # 連結されたダイアログ
    dialogue = "\n".join(dialogue_parts)

    # 候補リストとスコアリストを作成
    candidates = [entry["detail"] for entry in data["candidates"]]
    mentioned_id = data["mentioned"]["id"]
    score = [1 if entry["id"] == mentioned_id else 0 for entry in data["candidates"]]

    # 変換後のデータを返す
    return {
        "dialogue": dialogue,
        "candidates": candidates,
        "score": score
    }

# ディレクトリ内のすべてのJSONファイルを処理
input_dir = '../data/formatted_data'
output_dir = '../data/datasets_1'
# 出力ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

# すべてのJSONファイルを処理
for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith('.json'):
        input_file_path = os.path.join(input_dir, filename)
        output_file_path = os.path.join(output_dir, filename)
        
        # JSONファイルを読み込む
        with open(input_file_path, 'r', encoding='utf-8') as file:
            datas = json.load(file)

        
        transformed_datas = [transform_data(data) for data in datas]
        
        # 変換したデータをoutput_dirに同じファイル名で書き出す
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(transformed_datas, file, ensure_ascii=False, indent=4)

print("処理が完了しました。")