import json
import os
import glob

# Set base directory for path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_last_dialogue(directory_path, target_path):
    # ディレクトリ内のすべてのファイルを検索
    file_paths = glob.glob(os.path.join(directory_path, "*.json"))
    target_file_paths = glob.glob(os.path.join(target_path, "*.json"))
    target_file_names = {os.path.basename(path) for path in target_file_paths}

    # ファイルパスを辞書順（アルファベット順）にソート
    file_paths.sort()

    # 結果を保存するリスト
    results = []

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        if file_name in target_file_names:
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                    # データが配列形式の場合は最後の要素を取得
                    if isinstance(data, list):
                        if data:  # リストが空でないことを確認
                            last_dialogue = data[-1].get("dialogue", "")
                        else:
                            last_dialogue = ""
                    # データが辞書形式で直接dialogueキーを持つ場合
                    elif isinstance(data, dict):
                        last_dialogue = data.get("dialogue", "")
                    else:
                        last_dialogue = ""

                    # 結果をリストに追加
                    results.append({"file_name": file_name, "dialogue": last_dialogue})
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # 結果をファイル名で辞書順にソート
    results.sort(key=lambda x: x["file_name"])

    return results


def save_to_csv(results, output_path):
    import csv

    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # ヘッダー行を書き込む
        writer.writerow(["file_name", "dialogue"])

        # データ行を書き込む
        for result in results:
            writer.writerow([result["file_name"], result["dialogue"]])

    print(f"CSV file saved at {output_path}")


def print_results(results):
    for result in results:
        print(f"File Name: {result['file_name']}")
        print(f"Dialogue: {result['dialogue']}")
        print("-" * 50)


# メイン処理
def main():
    directory_path = os.path.join(BASE_DIR, "../../data/Tabidachi/datasets_1/")  # 処理するディレクトリパス
    target_path = os.path.join(BASE_DIR, "../../data/Tabidachi/cloudworker-dataset-proposal")
    output_path = os.path.join(BASE_DIR, "../../data/Tabidachi/extracted_last_dialogues.csv")  # 出力CSVファイルのパス

    results = extract_last_dialogue(directory_path, target_path)
    save_to_csv(results, output_path)

    # 結果を画面に表示（オプション）
    # print_results(results)


if __name__ == "__main__":
    main()
