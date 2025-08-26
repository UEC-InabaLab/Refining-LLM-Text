import os

def count_files_in_directory(directory_path: str) -> int:
    """
    指定されたディレクトリ内のファイルの個数をカウントする関数。
    
    :param directory_path: ファイルをカウントするディレクトリのパス
    :return: ファイルの個数
    """
    try:
        # ディレクトリ内のエントリを取得し、ファイルのみをカウント
        return len([entry for entry in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, entry))])
    except FileNotFoundError:
        print(f"エラー: 指定されたディレクトリ '{directory_path}' が存在しません。")
        return 0
    except PermissionError:
        print(f"エラー: 指定されたディレクトリ '{directory_path}' にアクセスできません。")
        return 0

if __name__=="__main__":
    # 使用例
    directory_path = "../data_ChatRec/datasets_3/Train-except_for_travel"  # ここに対象のディレクトリパスを指定
    file_count = count_files_in_directory(directory_path)
    print(f"ディレクトリ内のファイル数: {file_count}")