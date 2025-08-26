"""
metrics_sentence_morph.py
-------------------------
MeCab(=fugashi) 形態素で

  1. 要約文・推薦文の平均文字数
  2. 推薦文 vs. 説明文  BLEU / ROUGE
  3. 要約文・推薦文の Distinct-1 / Distinct-2

を一括計算します。  
行数（データ件数）も最後に表示します。

先に:
    pip install fugashi[unidic-lite] sacrebleu rouge-score pandas tqdm
必要なら `unidic` などに差し替えてください。
"""

import sys
import subprocess
import math, collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

# ---------------- 1. MeCab (fugashi) トークナイザー ----------------
try:
    from fugashi import Tagger
    tagger = Tagger()                       # デフォルト辞書
except Exception as e:
    print("⚠  Tagger() 初期化に失敗しました:", e)
    print("   → unidic-lite をインストールして再試行します …")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "unidic-lite"]
    )
    from fugashi import Tagger
    tagger = Tagger()
print("✓ MeCab tokenizer ready\n")

def tokenize(text: str):
    """形態素表層形のリスト"""
    return [tok.surface for tok in tagger(text)]


# ---------------- 2. Excel 読み込み ----------------
XLSX  = "cloud_worker_tabidachi_datasets.xlsx"
SHEET = "data"       # ← 変える場合はここ
df = pd.read_excel(XLSX, sheet_name=SHEET)
n_rows = len(df)
print(f"Loaded {n_rows} rows from sheet '{SHEET}'\n")

summary_cols = {
    "baseline": "dialogue_summary(SumRec)",
    "proposed": "dialogue_summary(提案手法)",
}
rec_cols = {
    "baseline": "recommend_sentence(SumRec)",
    "proposed": "recommend_sentence(提案手法)",
}
explanation_col = "candidate_info"


# ---------------- 3. 指標計算ユーティリティ ----------------
def distinct(texts, n=1):
    total, uniq = 0, set()
    for t in texts:
        toks = tokenize(str(t))
        grams = [" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)]
        uniq.update(grams)
        total += max(len(toks)-n+1, 0)
    return len(uniq)/total if total else 0.0

bleu_metric = BLEU()
def bleu_sentence(h, r):
    return bleu_metric.sentence_score(" ".join(tokenize(h)),
                                      [" ".join(tokenize(r))]).score

rouge_metric = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=False
)
def rouge_sentence(h, r):
    scores = rouge_metric.score(" ".join(tokenize(r)),
                                " ".join(tokenize(h)))
    return {k: v.fmeasure for k, v in scores.items()}

def avg_len(series):
    return series.astype(str).str.len().mean()


# ---------------- 4. 統計計算 ----------------
length_stats   = {}   # 平均文字数
distinct_stats = {}   # Dist-1/2
bleu_avg       = {}   # 推薦文 BLEU
rouge_avg      = {}   # 推薦文 ROUGE

# 4-A: 要約・推薦の文字数
for lbl, col in summary_cols.items():
    length_stats[f"summary_{lbl}"] = avg_len(df[col])
for lbl, col in rec_cols.items():
    length_stats[f"recommend_{lbl}"] = avg_len(df[col])

# 4-B: Distinct-1/2
for lbl, col in summary_cols.items():
    texts = df[col].fillna("")
    distinct_stats[f"summary_{lbl}"] = (
        distinct(texts, 1), distinct(texts, 2)
    )
for lbl, col in rec_cols.items():
    texts = df[col].fillna("")
    distinct_stats[f"recommend_{lbl}"] = (
        distinct(texts, 1), distinct(texts, 2)
    )

# 4-C: BLEU / ROUGE (推薦文 vs 説明文)
for lbl, col in rec_cols.items():
    b_list, r1, r2, rl = [], [], [], []
    for hyp, ref in tqdm(zip(df[col].fillna(""),
                             df[explanation_col].fillna("")),
                         total=n_rows,
                         desc=f"BLEU/ROUGE for {lbl}"):
        b_list.append(bleu_sentence(hyp, ref))
        r = rouge_sentence(hyp, ref)
        r1.append(r["rouge1"]); r2.append(r["rouge2"]); rl.append(r["rougeL"])
    bleu_avg[lbl]  = np.mean(b_list)
    rouge_avg[lbl] = dict(R1=np.mean(r1),
                          R2=np.mean(r2),
                          RL=np.mean(rl))


# ---------------- 5. 出力 ----------------
print("\n◆ 平均文字数 (chars)")
for k, v in length_stats.items():
    print(f"{k:<22}: {v:.1f}")

print("\n◆ Distinct-1 / Distinct-2 (morph)")
for k, (d1, d2) in distinct_stats.items():
    print(f"{k:<22}: D1={d1:.4f}  D2={d2:.4f}")

print("\n◆ BLEU & ROUGE (推薦文 vs 説明文)")
for lbl in rec_cols:
    r = rouge_avg[lbl]
    print(f"{lbl:<10}: BLEU={bleu_avg[lbl]:.4f}  "
          f"R1={r['R1']:.3f}  R2={r['R2']:.3f}  RL={r['RL']:.3f}")

print(f"\nProcessed {n_rows} rows ✔")