# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
# from datasets import Dataset
# from trl import DPOTrainer, DPOConfig
# import os
# from accelerate import Accelerator

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ['HF_HOME'] = '../.venv/cache'
# def log_memory_usage(stage=""):
#     print(f"=== Memory Usage at {stage} ===")
#     for i in range(torch.cuda.device_count()):
#         print(f"GPU {i}:")
#         print(f"  Allocated: {torch.cuda.memory_allocated(i)/1024**3:.2f} GiB")
#         print(f"  Reserved: {torch.cuda.memory_reserved(i)/1024**3:.2f} GiB")
#         print(f"  Free: {torch.cuda.memory_reserved(i)/1024**3 - torch.cuda.memory_allocated(i)/1024**3:.2f} GiB")
#     print("==============================")


# # 1. モデルとトークナイザーの読み込み
# tokenizer = AutoTokenizer.from_pretrained("tokyotech-llm/Llama-3.1-Swallow-8B-v0.1")
#     # pad_tokenをeos_tokenに設定
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# policy_model = AutoModelForCausalLM.from_pretrained(
#   "tokyotech-llm/Llama-3.1-Swallow-8B-v0.1", 
#     torch_dtype=torch.bfloat16
# )
# policy_model.gradient_checkpointing_enable()
# # 1.5 教師モデル（参照モデル）の読み込み
# reference_model = AutoModelForCausalLM.from_pretrained(
#     "tokyotech-llm/Llama-3.1-Swallow-8B-v0.1", 
#     torch_dtype=torch.bfloat16,
# )
# reference_model.eval()
# for param in reference_model.parameters():
#     param.requires_grad = False

# reference_model.gradient_checkpointing_enable()

# # 2. データセットの作成
# data = {
#     'prompt': [
#         '今日はどんな天気ですか？',
#         '自己紹介をお願いします。',
#         '好きな食べ物は何ですか？',
#         '趣味は何ですか？',
#         '旅行でおすすめの場所は？',
#         '最近読んだ本について教えてください。',
#         'プログラミング言語のPythonについてどう思いますか？',
#         'AIの将来についてどう考えますか？',
#         '健康を維持するためのアドバイスはありますか？',
#         '音楽のおすすめを教えてください。'
#     ],
#     'chosen': [
#         '今日は晴れていて、気持ちの良い天気です。',
#         '私はAIアシスタントで、皆さんのお役に立つ情報を提供します。',
#         '寿司が大好きです。特にマグロが好きです。',
#         '読書と散歩が趣味です。',
#         '京都は歴史的な場所が多く、おすすめです。',
#         '最近、「海辺のカフカ」を読みました。とても面白かったです。',
#         'Pythonはシンプルで使いやすい言語で、多くの用途に適しています。',
#         'AIはこれからも進化し、多くの分野で活躍すると考えています。',
#         'バランスの取れた食事と適度な運動が大切です。',
#         '最近はクラシック音楽にハマっています。'
#     ],
#     'rejected': [
#         'さあ、知りません。',
#         '秘密です。',
#         '食べ物は嫌いです。',
#         '趣味はありません。',
#         'どこにも行かない方がいいです。',
#         '本は読みません。',
#         'Pythonは嫌いです。',
#         'AIは怖いものです。',
#         '健康なんて気にしなくていいです。',
#         '音楽は聴きません。'
#     ]
# }

# data["prompt"] = data["prompt"] 
# data["chosen"] = data["chosen"] 
# data["rejected"] = data["rejected"] 
# dataset = Dataset.from_dict(data)
# log_memory_usage("after loading data")
# accelerator = Accelerator()
# policy_model, reference_model, dataset = accelerator.prepare(policy_model, reference_model, dataset)
# log_memory_usage("after accelerator.prepare")

# # 3. DPO設定
# dpo_config = DPOConfig(
#     beta=0.1,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     learning_rate=5e-5,
#     logging_steps=1,
#     save_steps=5,
#     save_total_limit=2,
#     num_train_epochs=1,
#     gradient_checkpointing=True,
#     bf16=True,
#     output_dir="./dpo-results",
#     remove_unused_columns=False,
# )
# log_memory_usage("after DPOConfig")

# # 4. DPOトレーナーの初期化と実行
# trainer = DPOTrainer(
#      model=policy_model.module if isinstance(policy_model, torch.nn.parallel.DistributedDataParallel) else policy_model,
#     ref_model=reference_model,
#     args=dpo_config,
#     train_dataset=dataset,
#     tokenizer=tokenizer,
# )
# log_memory_usage("after DPOTrainer")


# # 5. 学習の実行
# trainer.train()

# # 6. モデルの保存
# trainer.save_model("./dpo-results")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
import os
from accelerate import Accelerator

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['HF_HOME'] = '../.venv/cache'
def log_memory_usage(stage=""):
    print(f"=== Memory Usage at {stage} ===")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:")
        print(f"  Allocated: {torch.cuda.memory_allocated(i)/1024**3:.2f} GiB")
        print(f"  Reserved: {torch.cuda.memory_reserved(i)/1024**3:.2f} GiB")
        print(f"  Free: {torch.cuda.memory_reserved(i)/1024**3 - torch.cuda.memory_allocated(i)/1024**3:.2f} GiB")
    print("==============================")


# 1. モデルとトークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained("tokyotech-llm/Llama-3.1-Swallow-8B-v0.1")
    # pad_tokenをeos_tokenに設定
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
policy_model = AutoModelForCausalLM.from_pretrained(
  "tokyotech-llm/Llama-3.1-Swallow-8B-v0.1", 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
policy_model.gradient_checkpointing_enable()
# 1.5 教師モデル（参照モデル）の読み込み
reference_model = AutoModelForCausalLM.from_pretrained(
    "tokyotech-llm/Llama-3.1-Swallow-8B-v0.1", 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
reference_model.eval()
for param in reference_model.parameters():
    param.requires_grad = False

reference_model.gradient_checkpointing_enable()

# 2. データセットの作成
data = {
    'prompt': [
        '今日はどんな天気ですか？',
        '自己紹介をお願いします。',
        '好きな食べ物は何ですか？',
        '趣味は何ですか？',
        '旅行でおすすめの場所は？',
        '最近読んだ本について教えてください。',
        'プログラミング言語のPythonについてどう思いますか？',
        'AIの将来についてどう考えますか？',
        '健康を維持するためのアドバイスはありますか？',
        '音楽のおすすめを教えてください。'
    ],
    'chosen': [
        '今日は晴れていて、気持ちの良い天気です。',
        '私はAIアシスタントで、皆さんのお役に立つ情報を提供します。',
        '寿司が大好きです。特にマグロが好きです。',
        '読書と散歩が趣味です。',
        '京都は歴史的な場所が多く、おすすめです。',
        '最近、「海辺のカフカ」を読みました。とても面白かったです。',
        'Pythonはシンプルで使いやすい言語で、多くの用途に適しています。',
        'AIはこれからも進化し、多くの分野で活躍すると考えています。',
        'バランスの取れた食事と適度な運動が大切です。',
        '最近はクラシック音楽にハマっています。'
    ],
    'rejected': [
        'さあ、知りません。',
        '秘密です。',
        '食べ物は嫌いです。',
        '趣味はありません。',
        'どこにも行かない方がいいです。',
        '本は読みません。',
        'Pythonは嫌いです。',
        'AIは怖いものです。',
        '健康なんて気にしなくていいです。',
        '音楽は聴きません。'
    ]
}

data["prompt"] = data["prompt"] 
data["chosen"] = data["chosen"] 
data["rejected"] = data["rejected"] 
dataset = Dataset.from_dict(data)
log_memory_usage("after loading data")
accelerator = Accelerator()
log_memory_usage("after accelerator.prepare")

# 3. DPO設定
dpo_config = DPOConfig(
    beta=0.1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    logging_steps=1,
    save_steps=5,
    save_total_limit=2,
    num_train_epochs=1,
    gradient_checkpointing=True,
    bf16=True,
    output_dir="./dpo-results",
    remove_unused_columns=False,
)
log_memory_usage("after DPOConfig")

# 4. DPOトレーナーの初期化と実行
trainer = DPOTrainer(
     model=policy_model.module if isinstance(policy_model, torch.nn.parallel.DistributedDataParallel) else policy_model,
    ref_model=reference_model,
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
log_memory_usage("after DPOTrainer")


# 5. 学習の実行
trainer.train()

# 6. モデルの保存
trainer.save_model("./dpo-results")