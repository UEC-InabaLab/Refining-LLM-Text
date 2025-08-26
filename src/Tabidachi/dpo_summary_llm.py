import json
import os
import gc
from itertools import groupby
from tqdm import tqdm
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
import optuna
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
    103,
    104,
    105,
    106,
    107,
    109,
    110,
    111,
    112,
    113,
    115,
    116,
    117,
    118,
    121,
    122,
    123,
    124,
    125,
    201,
    202,
    204,
    205,
    207,
    208,
    210,
    302,
    303,
    304,
    306,
    307,
    308,
    309,
    310,
    311,
    313,
    314,
    315,
    318,
    319,
    320,
]
TEST_ID = [102, 114, 119, 203, 209, 305, 312, 316]
VALID_ID = [108, 120, 206, 301, 317]


def print_gpu_memory():
    print("====")
    print(f"利用可能なGPUの数: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  メモリ使用量: {torch.cuda.memory_allocated(i)/1024**3:.2f} GiB")
        print(
            f"  メモリキャップ: {torch.cuda.get_device_properties(i).total_memory/1024**3:.2f} GiB"
        )
    print("====")


# ----------------------------------------
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
    # input_dir = os.path.join(BASE_DIR, "../../data/ChatRec/datasets_3/")
    input_dir = os.path.join(BASE_DIR, "../../data/ChatRec/datasets_3/")  # 再学習時用

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


def create_valid_dataset() -> Dataset:
    # input_dir = os.path.join(BASE_DIR, "../../data/ChatRec/datasets_3/")
    input_dir = os.path.join(BASE_DIR, "../../data/ChatRec/datasets_3/")
    # 出力ディレクトリが存在しない場合は作成
    train_data = []
    for filename in tqdm(os.listdir(input_dir)):
        if int(filename[:3]) in VALID_ID:
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
    print(f'valid:{len(formatted_data["prompt"])}')
    train_dataset = Dataset.from_dict(formatted_data)
    return train_dataset


###########################################
# OptunaのObjective関数（検証データでの評価を使用）
###########################################
def objective(trial):
    # トライアル開始時のメモリクリア
    torch.cuda.empty_cache()
    gc.collect()

    # ハイパーパラメータの提案
    beta = trial.suggest_loguniform("beta", 0.01, 1.0)
    per_device_train_batch_size = trial.suggest_categorical(
        "per_device_train_batch_size", [8]
    )
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-8, 1e-6)
    num_train_epochs = trial.suggest_categorical(
        "num_train_epochs", [1]
    )  # 短いエポック数で評価

    # DPOの設定
    dpo_config = DPOConfig(
        beta=beta,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        logging_steps=1,
        save_steps=5,
        save_total_limit=2,
        num_train_epochs=num_train_epochs,  # 短いエポック数で評価
        gradient_checkpointing=True,
        bf16=True,
        output_dir=f"./dpo-summary-checkpoint",  # 各試行ごとに上書きされる可能性あり
        remove_unused_columns=False,
        report_to="wandb",
    )

    # トレーナーの初期化
    trainer = DPOTrainer(
        model=(
            policy_model.module
            if isinstance(policy_model, torch.nn.parallel.DistributedDataParallel)
            else policy_model
        ),
        ref_model=reference_model,
        args=dpo_config,
        beta=beta,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
    )

    # 学習実行
    try:
        trainer.train()
    except Exception as e:
        print(f"Trial failed with error: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        raise e

    metrics = trainer.evaluate()
    validation_score = metrics.get("eval_loss")
    print(f"Trial: beta={beta}, lr={learning_rate}, score={validation_score}")

    torch.cuda.empty_cache()
    gc.collect()

    return validation_score


###########################################
# メイン処理
###########################################
if __name__ == "__main__":
    wandb.init(dir="/tmp/wandb")  # dirを指定
    torch.cuda.empty_cache()
    gc.collect()

    # Accelerateの初期化（分散学習のための設定）
    # accelerator = Accelerator()
    # print(f"Accelerator状態: {accelerator.state}")

    # モデルとトークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained("tokyotech-llm/Llama-3.1-Swallow-8B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # policy_model はメインの学習対象として読み込む
    policy_model = AutoModelForCausalLM.from_pretrained(
        "tokyotech-llm/Llama-3.1-Swallow-8B-v0.1",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    policy_model.config.use_cache = False

    # 参照モデルは固定パラメータとして使用
    reference_model = AutoModelForCausalLM.from_pretrained(
        "tokyotech-llm/Llama-3.1-Swallow-8B-v0.1",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    reference_model.config.use_cache = False
    reference_model.config.pad_token_id = tokenizer.pad_token_id
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    # 学習用データと検証用データの作成
    train_dataset = create_train_dataset()
    valid_dataset = create_valid_dataset()

    # Optunaによるハイパーパラメータ探索
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    # ベストパラメータの表示
    best_params = study.best_params
    print("Best Parameters:", best_params)

    # ベストパラメータで再学習
    dpo_config = DPOConfig(
        beta=best_params["beta"],
        per_device_train_batch_size=best_params["per_device_train_batch_size"],
        gradient_accumulation_steps=1,
        learning_rate=best_params["learning_rate"],
        logging_steps=1,
        save_steps=5,
        save_total_limit=2,
        num_train_epochs=1,  # 最終モデルのエポック数（必要に応じて調整）
        gradient_checkpointing=True,
        bf16=True,
        output_dir="./dpo-summary-checkpoint",
        remove_unused_columns=False,
        report_to="wandb",
    )
    trainer = DPOTrainer(
        model=policy_model,
        ref_model=reference_model,
        args=dpo_config,
        beta=best_params["beta"],
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # ここでは，trainer.train()実行後に内部で記録されたログを利用してTensorBoardへ記録する
    trainer.train()

    # モデルの保存
    trainer.save_model("./dpo-summary-results_1")
