import json
import csv
import os
import glob


def process_files(directory_path):
    # ディレクトリ内のすべてのファイルを検索
    file_paths = glob.glob(os.path.join(directory_path, "*.json"))

    # ファイルパスを辞書順（アルファベット順）にソート
    file_paths.sort()

    # 結果を保存するリスト
    results = []

    for file_path in file_paths:
        file_name = os.path.basename(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

                dialogue_summary = data.get("dialogue_summary", "")

                # true_score が 1 のデータのみを抽出
                positive_candidates = []
                for candidate in data.get("candidates", []):
                    if candidate.get("true_score") == 1:
                        positive_candidates.append(
                            {
                                "candidate_info": candidate.get("candidate_info", ""),
                                "recommend_sentence": candidate.get(
                                    "recommend_sentence", ""
                                ),
                            }
                        )

                if positive_candidates:
                    # 結果をリストに追加
                    result = {
                        "file_name": file_name,
                        "dialogue_summary": dialogue_summary,
                        "positive_candidates": positive_candidates,
                    }
                    results.append(result)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return results


def save_to_csv(results, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # 最大の positive_candidates の数を取得
        max_candidates = (
            max([len(result["positive_candidates"]) for result in results])
            if results
            else 0
        )

        # ヘッダー行の作成
        headers = ["file_name", "dialogue_summary"]
        for i in range(max_candidates):
            headers.extend([f"candidate_info_{i+1}", f"recommend_sentence_{i+1}"])

        # ヘッダー行を書き込む
        writer.writerow(headers)

        # データ行を書き込む
        for result in results:
            file_name = result["file_name"]
            dialogue_summary = result["dialogue_summary"]

            # CSVの1行に必要なすべてのデータを収集
            row = [file_name, dialogue_summary]

            # すべての positive_candidates からデータを追加
            for candidate in result["positive_candidates"]:
                row.append(candidate["candidate_info"])
                row.append(candidate["recommend_sentence"])

            # CSVに行を書き込む
            writer.writerow(row)

    print(f"CSV file saved at {output_path}")


# メイン処理
def main():
    directory_path = "../data/cloudworker-dataset-baseline1"  # 処理するディレクトリパス
    output_path = "extracted_positive_candidates.csv"  # 出力CSVファイルのパス

    results = process_files(directory_path)
    save_to_csv(results, output_path)


if __name__ == "__main__":
    main()
