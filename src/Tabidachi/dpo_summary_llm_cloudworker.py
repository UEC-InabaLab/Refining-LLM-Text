import os
import io
import zipfile
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
from tqdm import tqdm
from itertools import groupby
import json
import re
import wandb

# Set base directory for path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------
# GPUメモリの断片化対策：プロセス開始前の環境変数設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "../../.venv/cache")

# ----------------------------------------
# IDリスト
TRAIN_ID = [
    101,
    104,
    105,
    106,
    109,
    110,
    112,
    113,
    115,
    117,
    118,
    121,
    122,
    123,
    124,
    201,
    202,
    204,
    205,
    208,
    210,
    303,
    304,
    306,
    307,
    309,
    311,
    313,
    314,
    315,
    318,
    320,
]
TEST_ID = [
    102,
    103,
    107,
    111,
    114,
    116,
    119,
    125,
    203,
    207,
    209,
    302,
    305,
    308,
    310,
    312,
    316,
    319,
]
VALID_ID = [108, 120, 206, 301, 317]


def load_dpo_config_from_zip(zip_path):
    with open(zip_path, "rb") as f:
        binary_data = f.read()
        with zipfile.ZipFile(io.BytesIO(binary_data), "r") as zip_file:
            for file_name in zip_file.namelist():
                if file_name.endswith("data.pkl"):
                    with zip_file.open(file_name) as inner_file:
                        content = inner_file.read()
                        dpo_config = pickle.loads(content)

                        # 必要なら属性修正
                        if not hasattr(
                            dpo_config, "include_inputs_for_metrics"
                        ) and hasattr(dpo_config, "include_for_metrics"):
                            dpo_config.include_inputs_for_metrics = getattr(
                                dpo_config, "include_for_metrics"
                            )
                            delattr(dpo_config, "include_for_metrics")

                        return dpo_config
    raise FileNotFoundError("data.pkl が ZIP ファイル内に見つかりませんでした。")


# ----------------------------------------
# candidate_informationが重複している際は，chosenが最も長いもののみを残す
def process_data(data):
    def is_japanese_text(text):
        """
        日本語の割合を計算し、8割以上であればTrueを返す。
        """
        japanese_chars = re.findall(r"[ぁ-んァ-ン一-龥]", text)
        total_chars = len(text)
        if total_chars == 0:
            return False
        return len(japanese_chars) / total_chars >= 0.8

    # フィルタリング処理
    filtered_data = [
        entry
        for entry in data
        if len(entry["chosen_dialogue_summary"]) >= 30
        and is_japanese_text(entry["chosen_dialogue_summary"])
    ]

    return filtered_data


def create_train_dataset() -> Dataset:
    # input_dir="../data/datasets_3/"
    input_dir = "../data/datasets_3/"  # new用
    # 出力ディレクトリが存在しない場合は作成
    train_data = []
    for filename in tqdm(os.listdir(input_dir)):
        if int(filename[:3]) in TRAIN_ID:
            with open(input_dir + filename, "r", encoding="utf-8") as file:
                datas = json.load(file)
                processed_datas = process_data(datas)
                train_data += processed_datas
    # データを整形
    formatted_data = {
        "prompt": [
            f"""
以下の内容をもとにBの観光地に対する趣味・経験を要約してください。
なるべく多くの情報を含む要約文を生成してください。

【要約元文章】
{entry["all_short_dialogue_summary"]}

【要約文章】
    
"""
            for entry in train_data
        ],
        "chosen": [entry["chosen_dialogue_summary"] for entry in train_data],
        "rejected": [entry["rejected_dialogue_summary"] for entry in train_data],
    }
    print(f'train:{len(formatted_data["prompt"])}')
    train_dataset = Dataset.from_dict(formatted_data)
    return train_dataset


if __name__ == "__main__":
    wandb.init(dir="/tmp/wandb")  # dirを指定
    # ZIPファイルのパス
    # zip_path = "./dpo-summary-results/training_args.bin"
    zip_path = "./dpo-summary-results_1/training_args.bin"  # newよう

    # DPOConfigのロード
    dpo_config = load_dpo_config_from_zip(zip_path)
    dpo_config.report_to = []
    dpo_config.logging_strategy = "no"
    dpo_config.save_strategy = "no"
    # print("ロードしたDPOConfig (安全に属性を表示します):")
    # beta = None
    # per_device_train_batch_size = None
    # learning_rate = None
    # for attr in dir(dpo_config):
    #     # アンダースコア始まりの内部属性を除外
    #     if not attr.startswith("_"):
    #         try:
    #             value = getattr(dpo_config, attr)
    #             if attr == "beta":
    #                 beta = value
    #             elif attr == "per_device_train_batch_size":
    #                 per_device_train_batch_size = value
    #             elif attr == "learning_rate":
    #                 learning_rate = value
    #         except AttributeError:
    #             continue

    # トークナイザーとモデルのロード
    tokenizer = AutoTokenizer.from_pretrained("tokyotech-llm/Llama-3.1-Swallow-8B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    output_filepath = f"./dpo-summary-results_cloudworker"
    policy_model = AutoModelForCausalLM.from_pretrained(
        "tokyotech-llm/Llama-3.1-Swallow-8B-v0.1",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    policy_model.config.use_cache = False

    reference_model = AutoModelForCausalLM.from_pretrained(
        "tokyotech-llm/Llama-3.1-Swallow-8B-v0.1",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    # 学習用データと検証用データの作成
    train_dataset = create_train_dataset()

    # DPOトレーナーの初期化
    trainer = DPOTrainer(
        model=policy_model,
        ref_model=reference_model,
        args=dpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # 学習の実行
    trainer.train()

    # モデルの保存
    trainer.save_model(output_filepath)
    print(f"モデルが保存されました。")
