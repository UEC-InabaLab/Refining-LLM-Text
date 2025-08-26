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
#----------------------------------------
# GPUメモリの断片化対策：プロセス開始前の環境変数設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['HF_HOME'] = '../.venv/cache'

#----------------------------------------

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
    def is_japanese_text(text):
        """
        日本語の割合を計算し、8割以上であればTrueを返す。
        """
        japanese_chars = re.findall(r'[ぁ-んァ-ン一-龥]', text)
        total_chars = len(text)
        if total_chars == 0:
            return False
        return len(japanese_chars) / total_chars >= 0.8

    # フィルタリング処理
    filtered_data = [
        entry for entry in data
        if len(entry["chosen_dialogue_summary"]) >= 30 and is_japanese_text(entry["chosen_dialogue_summary"])
    ]

    return filtered_data


def create_train_dataset() -> Dataset:
    input_dir = "../data_ChatRec/datasets_3"
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
あなたは高性能な分析アシスタントです。以下の対話履歴を分析して、ユーザ{entry["user_name"]}の嗜好・経験・趣味を抽出し、簡潔に要約してください。
1. タスク: 対話履歴からユーザ{entry["user_name"]}の嗜好・経験・趣味を抽出
2. 出力形式: 文章で出力
3. 抽出すべき情報
- 好きな活動（スポーツ、芸術、音楽鑑賞など）
- 好きな場所や訪問したい場所
- 好きな食べ物・飲み物
- 好きなエンターテイメント（映画、テレビ、ゲーム、音楽など）
- 興味のある物事
- 収集している物やコレクション
- 日常的に行っている楽しみな活動
4. 抽出してはいけない情報
- 性格特性（親切、几帳面など）
- コミュニケーションスタイル
- 価値観や信念
- 対人関係の特徴
- 感情表現のパターン
- 思考プロセスの特徴

--Example--
【対話履歴】
A:よろしくお願いします。\nB:こちらこそよろしくお願いいたします。今年のゴールデンウィークはお出かけされましたか。\nA:あいにく緊急事態宣言エリアだったので外出できませんでした。どちらかお出かけされましたか？\nB:私もちょっと買い物に出かけたくらいでほとんどうちにいました。去年もこんな感じのゴールデンウィークだったような気がします。\nA:そうですよね。コロナ以降なかなか外出は難しいですよね。\nB:去年の終わり頃は、来年になったら収束しそうな気がしていましたが、全然だめですね。\nA:もうしばらくはこのままでしょうかね。収束したらどんなことがしたいですか？\nB:思う存分旅行に行ったり、博物館や美術館に行きたいですね。\nA:旅行いいですね。国内ですか？それとも海外ですか？\nB:まずは国内旅行に行きたいです。空気と緑がきれいな場所でのんびりしたい気分です。\nA:素敵ですね。自然が豊かなところだと、とてもリフレッシュできそうですし。\nB:そうですね、外出自粛が続くと精神的につらいですよね。コロナ後に行きたい場所とかってございますか。\nA:自粛が長いので、ディズニーランドなどのテーマパークや音楽フェスに行ってワイワイしたいですね。\nB:ディズニーランドもいいですね。そういえば去年からずっと一度も行っていません。\nA:開園していましたが、なかなかチケットも取れなくて。早く普通に行けるようになる日が待ち遠しいです。\nB:ディズニーランドって定期的に行かないとなんだか落ち着かなくなりますよね。子供がとっても行きたがってます。\nA:お子さんもディズニーお好きなんですね。美女と野獣のアトラクションも新しくできましたし、また楽しめそうですよね。\nB:そうですね、一日も早く収束してもらいたいですね。テーマパークってどうしても密になってしまいますよね。\nA:そうですよね。もうしばらく先になるかもしれませんが、コロナが明けまで頑張りたいですね。\nB:ワクチン接種が早く進むといいなと思います。うちは母がようやく一回目の接種を終えました。\nA:そうでしたか。無事に接種の予約も取れてよかったですね。これから私たちも早く接種できるといいですね。\n"

【要約】
コロナ禍で緊急事態宣言エリアに住んでおり、外出を控えており、コロナ収束後はディズニーランドなどのテーマパークや音楽フェスに行きたいと考えている。ディズニーランドに行く習慣があるようだが、コロナ禍でチケット入手が難しい状況に悩んでいる。また、美女と野獣のアトラクションなど、ディズニーの新しい施設に興味を持っている。自然が豊かな場所での活動もリフレッシュできると考えている。ワクチン接種に前向きな姿勢を示している。

--Let's begin!--
【対話履歴】
{entry["dialogue"]}

【ユーザ{entry["user_name"]}の要約】
                
    """ for entry in data_list
        ],
        "chosen": [entry["chosen_dialogue_summary"] for entry in data_list],
        "rejected": [entry["rejected_dialogue_summary"] for entry in data_list],
    }
    return Dataset.from_dict(formatted_data)


if __name__ == "__main__":
    wandb.init(dir="/tmp/wandb") #dirを指定
    # ZIPファイルのパス
    # zip_path = "./dpo-summary-results/training_args.bin"
    zip_path = "./dpo-summary-results/training_args.bin" #newよう

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
    for i in range(4,5): #通常は(4)
        output_filepath = f"./dpo-summary-results_{i+2}"
        if os.path.exists(output_filepath):
            print(f"File {output_filepath} already exists. Skipping.")
            continue
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