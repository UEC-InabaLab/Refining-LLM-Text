# =====optuna=====
import optuna
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error, r2_score
import torch.nn as nn
import json
import os
from tqdm import tqdm
from safetensors.torch import load_file 
import wandb
import socket


TRAIN_ID = [101, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 115, 116, 117, 118, 121, 122, 123, 124, 125, 201, 202, 204, 205, 207, 208, 210, 302, 303, 304, 306, 307, 308, 309, 310, 311, 313, 314, 315, 318, 319, 320]
TEST_ID = [102, 114, 119, 203, 209, 305, 312, 316]
VALID_ID = [108, 120, 206, 301, 317]

SERVER ="BARD"
TRAIN_MODEL = "baseline2" #proposal&baseline1 or baseline2
os.environ["WANDB_PROJECT"]=f"Deberta_{TRAIN_MODEL}"
BEST_MODEL_DIR = f"./deberta_best_model_{TRAIN_MODEL}_new"
TEST_OUTPUT_DIR = f"./deberta_{TRAIN_MODEL}_new"
METHOD_FLAG = False #ベースライン1と提案手法の場合はTrue，ベースライン2の場合はFalse

# カスタムデータセット
class RecommendationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # SEPトークンの取得
        sep = self.tokenizer.sep_token
        
        if METHOD_FLAG:
        # 3つの文章をSEPトークンで区切って連結(提案手法&baseline1)
            combined_text = f"{item['dialogue_summary']}{sep}{item['recommendation_sentence']}{sep}{item['candidate_information']}"
        else:
        # 2つの文章をSEPトークンで区切って連結(baseline2)
            combined_text = f"{item['dialogue_summary']}{sep}{item['candidate_information']}"
        
        inputs = self.tokenizer(
            combined_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs["labels"] = torch.tensor(item["score"], dtype=torch.float)
        return inputs

# モデルの定義
class DebertaRegressionModel(nn.Module):
    def __init__(self, pretrained_model_name):
        super(DebertaRegressionModel, self).__init__()
        self.base_model = AutoModel.from_pretrained(pretrained_model_name)
        self.regression_head = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, 1),
            nn.Sigmoid()  # ここで出力を0〜1に制限
        )

        # 線形層の初期化
        nn.init.xavier_uniform_(self.regression_head[0].weight)
        nn.init.zeros_(self.regression_head[0].bias)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.regression_head(cls_output)  # ここでシグモイドが適用される

        loss = None
        if labels is not None:
            weights = torch.tensor([10.0 if label > 0.5 else 1.0 for label in labels]).to(logits.device)
            loss_fn = nn.MSELoss(reduction="none")
            loss = (loss_fn(logits.squeeze(), labels) * weights).mean()

        return {"logits": logits, "loss": loss}


# データセットのロード
def load_dataset():
    input_dir = '../data/datasets_2'
    train_dataset, test_dataset, valid_dataset = [], [], []
    for filename in tqdm(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            if int(filename[:3]) in TRAIN_ID:
                train_dataset += json.load(f)
            elif int(filename[:3]) in TEST_ID:
                test_dataset += json.load(f)
            elif int(filename[:3]) in VALID_ID:
                valid_dataset += json.load(f)
    return train_dataset, test_dataset, valid_dataset

train_data, test_data, valid_data = load_dataset()

tokenizer = AutoTokenizer.from_pretrained("globis-university/deberta-v3-japanese-large")

train_dataset = RecommendationDataset(train_data, tokenizer)
valid_dataset = RecommendationDataset(valid_data, tokenizer)
test_dataset = RecommendationDataset(test_data, tokenizer)

# 評価関数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    mse = mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return {"eval_loss": mse, "mse": mse, "r2": r2}

# Optunaの目的関数
def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
    per_device_train_batch_size = trial.suggest_categorical("batch_size", [4])
    epoch = trial.suggest_categorical("epoch", [1, 4, 10])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-1) 
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epoch,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=weight_decay,
        logging_dir="./logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=learning_rate,
        report_to="wandb"
    )

    model = DebertaRegressionModel("globis-university/deberta-v3-japanese-large")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate(eval_dataset=valid_dataset)

    # 最良モデルを保存
    if trainer.state.best_metric == metrics["eval_loss"]:
        trainer.save_model(BEST_MODEL_DIR)

    return metrics["eval_loss"]

#======学習する際は以下のコメントアウトをすべて外す=======
if __name__ == "__main__":
    # Optunaによるハイパーパラメータ探索
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    # 最良ハイパーパラメータの取得
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # 最良モデルでテストデータを評価
    best_model = DebertaRegressionModel("globis-university/deberta-v3-japanese-large")

    # safetensorsから状態辞書をロード
    state_dict = load_file(os.path.join(BEST_MODEL_DIR, "model.safetensors"))
    best_model.load_state_dict(state_dict)

    # モデルを評価モードに設定（必要に応じて）
    best_model.eval()

    trainer = Trainer(
        model=best_model,
        args=TrainingArguments(
            output_dir=TEST_OUTPUT_DIR,
            per_device_eval_batch_size=16,
            logging_dir="./logs",
        ),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print("Test Results:", test_results)
    wandb.alert(
        title=f"deberta_{TRAIN_MODEL} 学習終了",
        text = f"サーバー名:{SERVER}" + socket.gethostname()
    )
    wandb.finish()