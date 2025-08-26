import os

def count_files_in_directory(directory_path: str) -> int:
    """
    指定されたディレクトリ内のファイルの個数をカウントする関数。
    
    :param directory_path: ファイルをカウントするディレクトリのパス
    :return: ファイルの個数
    """
    try:
        # ディレクトリ内のエントリを取得し、ファイルのみをカウント
        return len([
            entry for entry in os.listdir(directory_path) 
            if os.path.isfile(os.path.join(directory_path, entry))
        ])
    except FileNotFoundError:
        print(f"エラー: 指定されたディレクトリ '{directory_path}' が存在しません。")
        return 0
    except PermissionError:
        print(f"エラー: 指定されたディレクトリ '{directory_path}' にアクセスできません。")
        return 0

if __name__ == "__main__":
    # データが格納されているベースディレクトリ
    base_dir = "../data_ChatRec/experiment_chat_and_rec"

    # カウントしたい9つのサブディレクトリ
    directories = [
        "Train-except_for_travel",
        "Test-except_for_travel",
        "Valid-except_for_travel",
        "Train-travel",
        "Test-travel",
        "Valid-travel",
        "Train-no_restriction",
        "Test-no_restriction",
        "Valid-no_restriction"
    ]

    # 各ディレクトリ内のファイル数を表示
    for d in directories:
        dir_path = os.path.join(base_dir, d)
        file_count = count_files_in_directory(dir_path)
        print(f"{d} 内のファイル数: {file_count}")