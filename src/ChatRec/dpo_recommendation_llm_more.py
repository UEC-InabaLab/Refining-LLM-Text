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
import wandb

# Set base directory for path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#----------------------------------------
# GPUメモリの断片化対策：プロセス開始前の環境変数設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['HF_HOME'] = os.path.join(BASE_DIR, "../../.venv/cache")


def load_dpo_config_from_zip(zip_path):
    with open(zip_path, "rb") as f:
        binary_data = f.read()
        with zipfile.ZipFile(io.BytesIO(binary_data), 'r') as zip_file:
            for file_name in zip_file.namelist():
                if file_name.endswith("data.pkl"):
                    with zip_file.open(file_name) as inner_file:
                        content = inner_file.read()
                        dpo_config = pickle.loads(content)

                        # 必要なら属性修正
                        if not hasattr(dpo_config, 'include_inputs_for_metrics') and hasattr(dpo_config, 'include_for_metrics'):
                            dpo_config.include_inputs_for_metrics = getattr(dpo_config, 'include_for_metrics')
                            delattr(dpo_config, 'include_for_metrics')

                        return dpo_config
    raise FileNotFoundError("data.pkl が ZIP ファイル内に見つかりませんでした。")

#----------------------------------------
# candidate_informationが重複している際は，chosenが最も長いもののみを残す
def process_data(data):
    # candidate_informationでソート
    data_sorted = sorted(data, key=lambda x: x["candidate_information"])
    result = []
    for key, group in groupby(data_sorted, key=lambda x: x["candidate_information"]):
        group_list = list(group)
        longest = max(group_list, key=lambda x: len(x["chosen_recommendation_sentence"]))
        result.append(longest)
    return result

def create_train_dataset() -> Dataset:
    input_dir = os.path.join(BASE_DIR, "../../data/ChatRec/datasets_4")
    data_list = []
    
    # Trainから始まるフォルダを取得
    train_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)) and d.startswith('Train')]
    
    # 各Trainフォルダ内のファイルを処理
    for train_dir in train_dirs:
        train_dir_path = os.path.join(input_dir, train_dir)
        for filename in tqdm(os.listdir(train_dir_path), desc=f"Loading data from {train_dir}"):
            with open(os.path.join(train_dir_path, filename), 'r', encoding='utf-8') as file:
                datas = json.load(file)
                processed_datas = process_data(datas)
                data_list += processed_datas
    formatted_data = {
        "prompt": [
            f"""
あなたはプロの観光地推薦者です。
観光地情報をもとに観光地推薦文を生成してください。観光地推薦文は一文で出力をしてください。

--Example1--
【観光地情報】
札幌市内南東部に位置する全天候型屋内ドーム。北海道コンサドーレ札幌、北海道日本ハムファイターズのホームスタジアムで、サッカーや野球の試合のほか、スポーツ・コンサートなど各種イベントも行われる。買い物や食事が楽しめるショップ、見晴らしのいい展望台あり。イベントのない時には、札幌ドームの裏側を見学するドームツアーも開催。所要90分以上～ ベビーおすすめ キッズおすすめ 女子おすすめ 冬におすすめ 雨でもOK

【観光地推薦文】
全天候型屋内ドームであり、北海道コンサドーレ札幌、北海道日本ハムファイターズのホームスタジアム。サッカーや野球の試合のほか、スポーツ・コンサートなど各種イベントも行われ他、買い物や食事も楽しむことができる。見晴らしの良い展望台があり，良い景色を見ることができる。お子様連れの方や雨の日でも観光したい方におすすめの観光地です。

--Let's begin!--
【観光地情報】
{entry["candidate_information"]}

【観光地推薦文】
    """ for entry in data_list
        ],
        "chosen": [entry["chosen_recommendation_sentence"] for entry in data_list],
        "rejected": [entry["rejected_recommendation_sentence"] for entry in data_list],
    }
    return Dataset.from_dict(formatted_data)


if __name__ == "__main__":
    wandb.init(dir="/tmp/wandb") #dirを指定
    # ZIPファイルのパス
    zip_path = "./dpo-recommendation-results_1/training_args.bin"

    # DPOConfigのロード
    dpo_config = load_dpo_config_from_zip(zip_path)
    dpo_config.report_to = []
    dpo_config.logging_strategy="no"
    dpo_config.save_strategy="no"
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
    for i in range(4):
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
        trainer.save_model(f"./dpo-recommendation-results_{i+2}")
        print(f"モデルが保存されました。")