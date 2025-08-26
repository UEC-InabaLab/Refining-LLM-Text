import os
import json

# Set base directory for path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def process_data(data):
    """
    入力データ（JSON形式の辞書）を加工して、
    ・dialogueを発話者をA,Bに置換し、「A:～\nB:～」の連結文字列に変換
    ・各placeに対して、questionnaireのevaluationの値（A_Score,B_Score）を追加（3以上なら1、3未満なら0）
    した辞書を返す関数
    """
    # --- Dialogueの整形 ---
    # 発話者名はデータによって変わるため、最初に登場した順にA,Bと割り当てる
    dialogue_lines = []
    speaker_mapping = {}
    next_label = "A"
    for turn in data.get("dialogue", []):
        orig_speaker = turn.get("speaker", "")
        if orig_speaker not in speaker_mapping:
            # 2人以上の場合、仕様がA,Bのみの場合は最初の2名のみを対象とする（それ以降はBに統一するなど、用途に合わせて調整可能）
            if len(speaker_mapping) < 2:
                speaker_mapping[orig_speaker] = next_label
                next_label = "B" if next_label == "A" else "A"
            else:
                # すでに2名割り当て済みの場合、以降は既存のspeaker名を使うか、Bとして扱う
                speaker_mapping[orig_speaker] = "B"
        new_speaker = speaker_mapping[orig_speaker]
        dialogue_lines.append(f"{new_speaker}:{turn.get('utterance', '')}\n")
    dialogue_str = "".join(dialogue_lines)

    # --- 評価値の辞書作成 ---
    # questionnaireのキーは元の発話者名になっていると仮定し、speaker_mappingを利用してA,Bに振り分ける
    questionnaire = data.get("questionnaire", {})
    eval_A = {}
    eval_B = {}
    for orig_speaker, details in questionnaire.items():
        if orig_speaker in speaker_mapping:
            mapped = speaker_mapping[orig_speaker]
            # 各評価は、該当するspeakerのevaluationリストから辞書を作成
            if mapped == "A":
                eval_A = {ev["id"]: ev["score"] for ev in details.get("evaluation", [])}
            elif mapped == "B":
                eval_B = {ev["id"]: ev["score"] for ev in details.get("evaluation", [])}

    # --- place の整形 ---
    places_output = []
    for place in data.get("place", []):
        pid = place.get("id", "")
        raw_a = eval_A.get(pid, 0)
        raw_b = eval_B.get(pid, 0)
        # 3以上なら1、3未満なら0に変換
        a_score = 1 if raw_a >= 3 else 0
        b_score = 1 if raw_b >= 3 else 0

        place_dict = {
            "description": place.get("description", ""),
            "name": place.get("name", ""),
            "A_Score": a_score,
            "B_Score": b_score
        }
        places_output.append(place_dict)

    result = {
        "dialogue": dialogue_str,
        "place": places_output
    }
    return result

def process_files(input_root=None, output_root=None, subdirs=["except_for_travel", "travel", "no_restriction"]):
    if input_root is None:
        input_root = os.path.join(BASE_DIR, "../../data/ChatRec/chat_and_rec")
    if output_root is None:
        output_root = os.path.join(BASE_DIR, "../../data/ChatRec/processed_data")
    """
    ../data_ChatRec/chan_and_rec 以下の各サブディレクトリ（except_for_travel, travel, no_restriction）内の
    JSONファイルを読み込み、process_data()で加工し、同じサブディレクトリ構造で
    ./data_ChatRec/processed_data 配下に出力する関数
    """
    for subdir in subdirs:
        input_dir = os.path.join(input_root, subdir)
        output_dir = os.path.join(output_root, subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in os.listdir(input_dir):
            input_path = os.path.join(input_dir, filename)
            if os.path.isfile(input_path) and filename.endswith(".json"):
                try:
                    with open(input_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"JSON読み込みエラー {input_path}: {e}")
                    continue
                
                processed_data = process_data(data)
                
                output_path = os.path.join(output_dir, filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=4)
                print(f"Processed: {input_path} -> {output_path}")

if __name__ == "__main__":
    process_files()