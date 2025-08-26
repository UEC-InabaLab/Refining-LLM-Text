import os
import shutil
import random

def split_and_copy_files(src_folder, dst_folder, train_count, test_count, valid_count):
    """
    src_folder 内のファイルをランダムシャッフルして
    train_count, test_count, valid_count の順に取得し、
    dst_folder下の Train-xxx, Test-xxx, Valid-xxx フォルダにコピーする。
    """
    # src_folder からすべてのファイルを取得
    all_files = os.listdir(src_folder)
    all_files = [f for f in all_files if os.path.isfile(os.path.join(src_folder, f))]

    # ランダムにシャッフル
    random.shuffle(all_files)

    # ファイルを切り分け
    train_files = all_files[:train_count]
    test_files = all_files[train_count:train_count + test_count]
    valid_files = all_files[train_count + test_count:train_count + test_count + valid_count]

    # カテゴリ名(フォルダ名)を取得(except_for_travel, travel, no_restriction)
    category_name = os.path.basename(src_folder)

    # コピー先のパスを作成
    train_dst = os.path.join(dst_folder, f"Train-{category_name}")
    test_dst = os.path.join(dst_folder, f"Test-{category_name}")
    valid_dst = os.path.join(dst_folder, f"Valid-{category_name}")

    # コピー先フォルダが存在しなければ作成
    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(test_dst, exist_ok=True)
    os.makedirs(valid_dst, exist_ok=True)

    # Trainファイルをコピー
    for f in train_files:
        shutil.copy2(os.path.join(src_folder, f), os.path.join(train_dst, f))

    # Testファイルをコピー
    for f in test_files:
        shutil.copy2(os.path.join(src_folder, f), os.path.join(test_dst, f))

    # Validファイルをコピー
    for f in valid_files:
        shutil.copy2(os.path.join(src_folder, f), os.path.join(valid_dst, f))

def main():
    # 乱数シードを固定したい場合は設定(任意)
    # random.seed(42)

    # 元データがあるフォルダ
    base_src_dir = "../data_ChatRec/processed_data"

    # 出力先のベースフォルダ
    base_dst_dir = "../data_ChatRec/experiment_chat_and_rec"

    # カテゴリごとの分割数を辞書にまとめておく
    # { カテゴリ名: (train, test, valid) }
    split_counts = {
        "except_for_travel": (178, 33, 12),
        "travel": (189, 35, 13),
        "no_restriction": (436, 81, 28)
    }

    # 上記カテゴリに対してファイルを分割してコピー
    for category, counts in split_counts.items():
        src_folder = os.path.join(base_src_dir, category)
        split_and_copy_files(
            src_folder=src_folder,
            dst_folder=base_dst_dir,
            train_count=counts[0],
            test_count=counts[1],
            valid_count=counts[2]
        )

if __name__ == "__main__":
    main()