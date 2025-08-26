# Refining Text Generation for Realistic Conversational Recommendation via Direct Preference Optimization

本リポジトリは、論文「Refining Text Generation for Realistic Conversational Recommendation via Direct Preference Optimization」の実装コードです。Direct Preference Optimization（DPO）を用いて会話型推薦システムのテキスト生成を改善する手法を提案しています。

## 📖 概要

![提案手法フロー](images/proposal_flow.png)
*[高解像度図表を見る（PDF）](images/proposal_flow.pdf)*

従来の会話型推薦システム（CRS）は、短時間での性急な推薦や暗黙的情報の不十分な統合といった課題がありました。本研究では、SumRecを拡張し、DPOを用いて対話要約文生成モデルと観光地推薦文生成モデルを最適化することで、より現実的で自然な会話型推薦を実現します。

### 主な貢献
1. DPOを用いたSumRecの拡張により、現実的な会話型推薦データセットに適した推薦手法を提案
2. ベースライン手法や元のSumRecとの比較を通じて、提案手法の優れた推薦性能を実証

## 🏗️ システム構成

### 2段階学習アプローチ
- **Stage 1**: スコア予測器（DeBERTa）の事前学習
- **Stage 2**: DPOによる対話要約・推薦文生成モデルの学習

### モデル構成
- **ベースモデル**: Llama-3.1-Swallow-8B-v0.1
- **スコア予測器**: DeBERTa-v3-japanese-large
- **最適化手法**: Direct Preference Optimization (DPO)

## 📊 データセット・モデル入手先

### データセット
- **Tabidachiコーパス**: https://www.nii.ac.jp/dsc/idr/rdata/Tabidachi/
  - 旅行代理店の観光地推薦対話（現実的な長い対話）
- **ChatRecデータセット**: https://github.com/Ryutaro-A/SumRec
  - 多カテゴリ推薦対話（比較用）

### 事前学習済みモデル
- **Llama-3.1-Swallow-8B-v0.1**: https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-v0.1
- **DeBERTa-v3-japanese-large**: https://huggingface.co/globis-university/deberta-v3-japanese-large

## 🛠️ 環境構築・セットアップ

### 必要環境
- Python 3.10.12
- GPU: Nvidia A100 80GB × 4台（推奨）
- CUDA対応環境

### 環境構築手順

```bash
# 仮想環境の作成
python -m venv .venv

# 環境の有効化
source .venv/bin/activate  # Linux/Mac
# または .venv\Scripts\activate  # Windows

# 依存関係のインストール
pip install -r requirements.txt
```

### 主要依存関係
- PyTorch 2.4.1
- Transformers 4.46.2
- TRL (Transformer Reinforcement Learning) 0.12.1
- Optuna 4.1.0
- Datasets 3.1.0
- その他詳細は `requirements.txt` を参照

## 📂 データセット構成

本プロジェクトでは4種類のデータセット形式を使用します：

- **datasets_1**: 推薦時のデータセット（対話要約文と観光地推薦文作成用）
- **datasets_2**: スコア予測器学習用
- **datasets_3**: 対話要約文生成モデルDPO用
- **datasets_4**: 観光地推薦文生成モデルDPO用

### データ分割

#### Tabidachiコーパス
- **分割方法**: 人（ユーザーID）ベース
- **データ構成**:
  - 訓練データ：成年20名、高齢者7名、子ども15名
  - 検証データ：成年2名、高齢者1名、子ども2名
  - テストデータ：成年3名、高齢者2名、子ども3名
- **ユーザーID範囲**:
  - 成年：101～125（25名）
  - 高齢者：201～210（10名）
  - 子ども：301～320（20名）
- **スコア付け**: 実際に推薦されたアイテムが1、それ以外が0

#### ChatRecコーパス
- **分割方法**: カテゴリベース
- **カテゴリ別データ数**:
  - except_for_travel：223個（訓練178、テスト33、検証12）
  - travel：237個（訓練189、テスト35、検証13）
  - no_restriction：545個（訓練436、テスト81、検証28）
- **スコア変換**: 
  - 人間予測スコア（5第三者ワーカーによる1-5点評価）→ スコア3未満を0（dislike）、スコア3以上を1（like）に変換
  - この変換によりTabidachiコーパスと同様の二値分類に統一

#### 共通事項
- **割合**: 訓練:テスト:検証 = 8:1.5:0.5

## 🚀 実行手順

### 1. データ前処理
```bash
# データセットの作成（順番に実行）
python src/create_dataset_1.py
python src/create_dataset_2.py
python src/create_dataset_3.py
python src/create_dataset_4.py
```

### 2. スコア予測器の学習
```bash
python src/train_deberta.py
```

### 3. DPO学習
```bash
# 対話要約文生成モデルのDPO学習
python src/dpo_summary_llm.py

# 推薦文生成モデルのDPO学習
python src/dpo_recommendation_llm.py
```

### 4. 推薦データ生成・評価
```bash
# 提案手法での推薦データ生成
python src/create_recommend_data_proposal.py

# ベースライン手法での推薦データ生成
python src/create_recommend_data_baseline1.py
python src/create_recommend_data_baseline2.py

# 評価実行
python src/evaluate_from_recommend_data.py
```

## 🔬 実験設定・評価

### 比較手法
- **Baseline**: 対話要約のみ使用、推薦文生成なし
- **SumRec**: DPO前の元手法
- **Proposed**: DPO適用済み提案手法

### アブレーション研究
- **w/o Rec-DPO**: 推薦文生成モデルのDPO未適用
- **w/o Sum-DPO**: 対話要約文生成モデルのDPO未適用

### 評価指標
- **Hit Rate (HR)@k**: 上位k件に正解が含まれる割合
- **Mean Reciprocal Rank (MRR)@k**: 正解アイテムの逆順位の平均
- k = 1, 3, 5で評価

### ハイパーパラメータ最適化
- Optunaを用いた自動最適化
- 最適化対象: 学習率、バッチサイズ、DPOのβパラメータ

## 📈 実験結果

### 主要結果
- **Tabidachiコーパス**: 全指標（HR@1,3,5, MRR@1,3,5）において既存手法を上回る性能
- **ChatRecコーパス**: MRRで一貫して最高性能を達成

### 重要な知見
1. **対話要約文のDPO学習が特に重要**: アブレーション研究により確認
2. **生成テキストの質的向上**: より詳細で推薦関連情報を重視したテキスト生成
3. **上位ランクでの改善**: 特にHR@1, MRR@1で顕著な性能向上

## 📁 ファイル構成

```
├── src/                          # メイン実装（Tabidachi用）
│   ├── create_dataset_*.py       # データセット作成スクリプト
│   ├── train_deberta.py          # スコア予測器学習
│   ├── dpo_summary_llm.py        # 対話要約文DPO学習
│   ├── dpo_recommendation_llm.py # 推薦文DPO学習
│   ├── create_recommend_data_*.py # 推薦データ生成
│   ├── evaluate_from_recommend_data.py # 評価スクリプト
│   ├── rec_model.py              # 推薦モデル
│   └── summary_model.py          # 要約モデル
├── src-chatrec/                  # ChatRec用実装
├── images/                       # README用画像
│   ├── proposal_flow.pdf         # 提案手法フロー図（PDF）
│   └── proposal_flow.png         # 提案手法フロー図（PNG）
├── metrics_sentence.py           # テキスト品質評価
├── requirements.txt              # 依存関係
└── README_JP.md                  # このファイル
```

### 主要スクリプトの説明

#### データ前処理
- `create_dataset_1.py`: 基本的な推薦データセット作成
- `create_dataset_2.py`: スコア予測器用データセット作成
- `create_dataset_3.py`: 対話要約DPO用選好データ作成
- `create_dataset_4.py`: 推薦文DPO用選好データ作成

#### モデル学習
- `train_deberta.py`: DeBERTaベーススコア予測器の学習
- `dpo_summary_llm.py`: Optunaを用いた対話要約文生成モデルのDPO学習
- `dpo_recommendation_llm.py`: Optunaを用いた推薦文生成モデルのDPO学習

#### 推薦・評価
- `create_recommend_data_proposal.py`: 提案手法での推薦実行
- `create_recommend_data_baseline*.py`: ベースライン手法での推薦実行
- `evaluate_from_recommend_data.py`: HR@k, MRR@k評価の実行

## 💻 実行時の注意事項

### GPU・メモリ要件
- 推奨: Nvidia A100 80GB × 4台
- Llama-3.1-Swallow-8Bの学習: 約24時間（4GPU使用）
- DeBERTaの学習: 約4時間（4GPU使用）

### 論文引用
**今後追記**


**Note**: 本実装は研究目的で作成されており、商用利用時は各モデル・データセットのライセンスを確認してください。