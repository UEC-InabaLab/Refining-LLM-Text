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

#----------------------------------------
# GPUメモリの断片化対策：プロセス開始前の環境変数設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['HF_HOME'] = '../.venv/cache'

#----------------------------------------
# IDリスト
TRAIN_ID = [101, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 115, 116, 117, 118, 121, 122, 123, 124, 125, 201, 202, 204, 205, 207, 208, 210, 302, 303, 304, 306, 307, 308, 309, 310, 311, 313, 314, 315, 318, 319, 320]
TEST_ID = [102, 114, 119, 203, 209, 305, 312, 316]
VALID_ID = [108, 120, 206, 301, 317]

def print_gpu_memory():
    print("====")
    print(f"利用可能なGPUの数: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  メモリ使用量: {torch.cuda.memory_allocated(i)/1024**3:.2f} GiB")
        print(f"  メモリキャップ: {torch.cuda.get_device_properties(i).total_memory/1024**3:.2f} GiB")
    print("====")

#----------------------------------------
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

def create_valid_dataset() -> Dataset:
    input_dir = "../data_ChatRec/datasets_3"
    data_list = []
    
    # Trainから始まるフォルダを取得
    train_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)) and d.startswith('Valid')]
    
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

###########################################
# OptunaのObjective関数（検証データでの評価を使用）
###########################################
def objective(trial):
    # トライアル開始時のメモリクリア
    torch.cuda.empty_cache()
    gc.collect()

    # ハイパーパラメータの提案
    beta = trial.suggest_loguniform("beta", 0.01, 1.0)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-8, 1e-6)
    num_train_epochs = trial.suggest_categorical("num_train_epochs", [1]) # 短いエポック数で評価
    
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
        report_to="wandb"
    )
    
    # トレーナーの初期化
    trainer = DPOTrainer(
        model=policy_model.module if isinstance(policy_model, torch.nn.parallel.DistributedDataParallel) else policy_model,
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
    wandb.init(dir="/tmp/wandb") #dirを指定
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
        report_to="wandb"
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
    trainer.save_model("./dpo-summary-results")