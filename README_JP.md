# Direct Preference Optimizationを用いた現実的な対話型推薦のためのテキスト生成の洗練

本リポジトリは、研究論文「Refining Text Generation for Realistic Conversational Recommendation via Direct Preference Optimization」の実装コードを含んでいます。Direct Preference Optimization (DPO)を用いて対話型推薦システム（CRS）のテキスト生成を改善する手法を提案しています。

## 📖 概要

![提案手法のフロー](images/proposal_flow.png)
*[高解像度図（PDF）を見る](images/proposal_flow.pdf)*

従来の対話型推薦システム（CRS）は、短い会話での性急な推薦や暗黙的情報の統合不足などの課題に直面しています。本研究では、対話要約生成とアイテム推薦情報生成モデルの両方を最適化するためにDPOを適用してSumRecを拡張し、より現実的で自然な対話型推薦を実現します。

### 主な貢献
1. DPOを用いてSumRecを拡張し、現実的な対話型推薦データセットに適した推薦手法を提案
2. ベースライン手法および元のSumRecとの比較により、優れた推薦性能を実証

## 🏗️ システムアーキテクチャ

### 2段階訓練アプローチ
- **ステージ1**: スコア予測器（DeBERTa）の事前訓練
- **ステージ2**: 対話要約生成とアイテム推薦情報生成モデルのDPO訓練

### モデル構成
- **ベースモデル**: Llama-3.1-Swallow-8B-v0.1
- **スコア予測器**: DeBERTa-v3-japanese-large
- **最適化手法**: Direct Preference Optimization (DPO)

## 📊 データセットとモデルソース

### データセット

#### Tabidachiコーパス
- **ソース**: https://www.nii.ac.jp/dsc/idr/rdata/Tabidachi/
- **説明**: 旅行代理店の推薦対話（現実的な長い会話）
- **ダウンロードと配置**:
  1. 上記リンクからデータセットをダウンロード
  2. ダウンロードしたファイルを `data/Tabidachi/annotation_data/` に配置
  
  **配置後のディレクトリ構造:**
  ```
  data/Tabidachi/
  └── annotation_data/
      ├── annotations/          # 対話アノテーションファイルを含むディレクトリ
      │   └── *.json           # 個別の対話セッションファイル
      ├── spot_info.json        # 観光地情報
      └── タグ一覧.docx         # タグ一覧ドキュメント
  ```

#### ChatRecデータセット
- **ソース**: https://github.com/Ryutaro-A/SumRec
- **説明**: 多カテゴリ推薦対話（比較用）
- **ダウンロードと配置**:
  1. GitHubリポジトリからデータセットをダウンロード
  2. ダウンロードしたファイルを `data/ChatRec/chat_and_rec/` に配置
  
  **配置後のディレクトリ構造:**
  ```
  data/ChatRec/
  └── chat_and_rec/
      ├── except_for_travel/    # 旅行以外の推薦対話
      │   └── *.json
      ├── travel/               # 旅行推薦対話
      │   └── *.json
      └── no_restriction/       # 一般推薦対話
          └── *.json
  ```

### 事前学習済みモデル
- **Llama-3.1-Swallow-8B-v0.1**: https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-v0.1
- **DeBERTa-v3-japanese-large**: https://huggingface.co/globis-university/deberta-v3-japanese-large

## 🛠️ 環境セットアップ

### 必要要件
- Python 3.10.12
- CUDA対応GPU環境

### 実験で使用したハードウェア
- GPU: 4 × Nvidia A100 80GB

### セットアップ手順

```bash
# 仮想環境の作成
python -m venv .venv

# 環境の有効化
source .venv/bin/activate  # Linux/Mac
# または .venv\Scripts\activate  # Windows

# 依存関係のインストール
pip install -r requirements.txt
```

### 主要な依存関係
- PyTorch 2.4.1
- Transformers 4.46.2
- TRL (Transformer Reinforcement Learning) 0.12.1
- Optuna 4.1.0
- Datasets 3.1.0
- 完全な詳細は `requirements.txt` を参照

## 📂 データセット構成

本プロジェクトでは4種類のデータセット形式を使用します：

- **datasets_1**: 推薦データセット（対話要約生成とアイテム推薦情報生成用）
- **datasets_2**: スコア予測器訓練データセット
- **datasets_3**: 対話要約生成モデル用DPOデータセット
- **datasets_4**: アイテム推薦情報生成モデル用DPOデータセット

### データ分割

#### Tabidachiコーパス
- **分割方法**: ユーザーIDベース
- **データ構成**:
  - 訓練データ: 大人20名、高齢者7名、子供15名
  - 検証データ: 大人2名、高齢者1名、子供2名
  - テストデータ: 大人3名、高齢者2名、子供3名
- **ユーザーID範囲**:
  - 大人: 101-125（25ユーザー）
  - 高齢者: 201-210（10ユーザー）
  - 子供: 301-320（20ユーザー）
- **スコアリング**: 実際に推薦されたアイテム = 1、その他 = 0

#### ChatRecコーパス
- **分割方法**: カテゴリベース
- **カテゴリ別データ数**:
  - except_for_travel: 223件（訓練: 178、テスト: 33、検証: 12）
  - travel: 237件（訓練: 189、テスト: 35、検証: 13）
  - no_restriction: 545件（訓練: 436、テスト: 81、検証: 28）
- **スコア変換**: 
  - 人間による予測スコア（5人の第三者による1-5段階評価）→ スコア<3を0（嫌い）、スコア≥3を1（好き）に変換
  - この変換により、Tabidachiコーパスの二値分類と統一

#### 共通仕様
- **比率**: 訓練:テスト:検証 = 8:1.5:0.5

## 🚀 実行ワークフロー

### A. Tabidachi実験

#### ステップ1: データセット準備
```bash
# Tabidachiデータセットをダウンロードして data/Tabidachi/annotation_data/ に配置
# その後、生データを前処理
python src/Tabidachi/data_preprocessing.py
```

#### ステップ2: 訓練データセットの作成
```bash
# すべてのデータセット形式を作成するために順番に実行
python src/Tabidachi/create_dataset_1.py  # 基本推薦データセット
python src/Tabidachi/create_dataset_2.py  # スコア予測器訓練データ
python src/Tabidachi/create_dataset_3.py  # 対話要約生成用DPOデータ
python src/Tabidachi/create_dataset_4.py  # アイテム推薦情報用DPOデータ
```

#### ステップ3: スコア予測器の訓練
```bash
# メソッド選択でDeBERTaを訓練
# メソッドオプション: proposal&baseline1, baseline2
# - proposal&baseline1: 3つの入力を使用（対話要約、アイテム推薦情報、候補情報）
# - baseline2: 2つの入力を使用（対話要約、候補情報）
python src/Tabidachi/train_deberta.py --method proposal&baseline1
# または:
python src/Tabidachi/train_deberta.py --method baseline2

# 出力: モデルは src/Tabidachi/deberta_best_model_[method]/ に保存
```

#### ステップ4: DPO訓練（Optunaを使用）
```bash
# ハイパーパラメータ最適化を使用して対話要約生成モデルを訓練
python src/Tabidachi/dpo_summary_llm.py         # モデル1を作成
python src/Tabidachi/dpo_summary_llm_more.py    # モデル2-5を作成
# 出力: モデルが dpo-summary-results_1/, dpo-summary-results_2/ などとして保存

# ハイパーパラメータ最適化を使用してアイテム推薦情報生成モデルを訓練
python src/Tabidachi/dpo_recommendation_llm.py      # モデル1を作成
python src/Tabidachi/dpo_recommendation_llm_more.py  # モデル2-5を作成
# 出力: モデルが dpo-recommendation-results_1/, dpo-recommendation-results_2/ などとして保存
```

#### ステップ5: 推薦の生成
```bash
# 提案手法（DPO訓練済みモデルを使用）
python src/Tabidachi/create_recommend_data_proposal.py

# 比較用ベースライン手法
python src/Tabidachi/create_recommend_data_baseline1.py  # baseline1 = SumRec（論文）
python src/Tabidachi/create_recommend_data_baseline2.py  # baseline2 = Baseline（論文）

# アブレーションスタディ（オプション）
python src/Tabidachi/create_recommend_data_ablation1.py  # Rec-DPOなし
python src/Tabidachi/create_recommend_data_ablation2.py  # Sum-DPOなし
```

#### ステップ6: 性能評価
```bash
# メソッド選択で評価
# proposal/ablationメソッド: 自動的にモデル1-5を評価して平均を計算
# baselineメソッド: 単一データセットを評価

# 提案手法（5つのモデルすべてを評価）
python src/Tabidachi/evaluate_from_recommend_data.py --method proposal

# ベースライン手法（それぞれ単一データセット）
python src/Tabidachi/evaluate_from_recommend_data.py --method baseline1
python src/Tabidachi/evaluate_from_recommend_data.py --method baseline2

# アブレーションスタディ（5つのモデルすべてを評価）
python src/Tabidachi/evaluate_from_recommend_data.py --method ablation1
python src/Tabidachi/evaluate_from_recommend_data.py --method ablation2

# 出力: 選択したメソッドのHR@kとMRR@kメトリクス
```

### B. ChatRec実験

#### ステップ1: データセット準備
```bash
# ChatRecデータセットをダウンロードして data/ChatRec/chat_and_rec/ に配置
# その後、生データを前処理
python src/ChatRec/data_preprocessing.py
```

#### ステップ2: 訓練データセットの作成
```bash
# 順番に実行
python src/ChatRec/create_dataset_1.py  # 基本推薦データセット
python src/ChatRec/create_dataset_2.py  # スコア予測器訓練データ
python src/ChatRec/create_dataset_3.py  # 対話要約生成用DPOデータ
python src/ChatRec/create_dataset_4.py  # アイテム推薦情報用DPOデータ
```

#### ステップ3: スコア予測器の訓練
```bash
# メソッド選択でDeBERTaを訓練
# メソッドオプション: proposal&baseline1, baseline2
# - proposal&baseline1: 3つの入力を使用（対話要約、アイテム推薦情報、候補情報）
# - baseline2: 2つの入力を使用（対話要約、候補情報）
python src/ChatRec/train_deberta.py --method proposal&baseline1
# または:
python src/ChatRec/train_deberta.py --method baseline2

# 出力: モデルは src/ChatRec/ChatRec_deberta_best_model_[method]/ に保存
```

#### ステップ4: DPO訓練
```bash
# DPO訓練
python src/ChatRec/dpo_summary_llm.py         # モデル1を作成
python src/ChatRec/dpo_summary_llm_more.py    # モデル2-5を作成
python src/ChatRec/dpo_recommendation_llm.py      # モデル1を作成
python src/ChatRec/dpo_recommendation_llm_more.py  # モデル2-5を作成
```

#### ステップ5: 推薦の生成と評価
```bash
# 推薦データの生成
python src/ChatRec/create_recommend_data_proposal.py    # 提案手法
python src/ChatRec/create_recommend_data_baseline1.py   # baseline1 = SumRec（論文）
python src/ChatRec/create_recommend_data_baseline2.py   # baseline2 = Baseline（論文）

# メソッド選択で性能を評価
# proposal: 自動的にモデル1-5を評価して平均を計算
# baselineメソッド: 単一データセットを評価

# 提案手法（5つのモデルすべてを評価）
python src/ChatRec/evaluate_from_recommend_data.py --method proposal

# ベースライン手法（それぞれ単一データセット）
python src/ChatRec/evaluate_from_recommend_data.py --method baseline1
python src/ChatRec/evaluate_from_recommend_data.py --method baseline2
```

### C. クラウドワーカー評価（Tabidachi）

#### ステップ1: 専用モデルの訓練
```bash
# クラウドワーカー評価用に異なるデータ分割でモデルを訓練
python src/Tabidachi/dpo_summary_llm_cloudworker.py
# 出力: モデルは src/Tabidachi/dpo-summary-results_cloudworker/ として保存
```

#### ステップ2: 評価データセットの作成
```bash
# ベースライン手法用（SumRec - DPOなし）
python src/Tabidachi/create_cloudworker_dataset.py --method baseline1
# 出力: data/Tabidachi/cloudworker-dataset-baseline1/

# 提案手法用（DPO訓練済みモデルを使用）
python src/Tabidachi/create_cloudworker_dataset.py --method proposal
# 出力: data/Tabidachi/cloudworker-dataset-proposal/

# 注: スクリプトは適切なモデルを自動選択：
# - baseline1: DPOなしのベースLlama-Swallowモデルを使用
# - proposal: DPO訓練済みモデルを使用（dpo-recommendation-results_1 と dpo-summary-results_cloudworker）
```

#### ステップ3: 人間評価の実施
1. 生成されたデータセットをクラウドワーカーに配布
2. 好み評価と定性的フィードバックを収集
3. 結果を `cloud_worker_tabidachi_datasets.xlsx` に保存

#### ステップ4: 結果の分析
```bash
# クラウドワーカー評価結果の分析
python metrics_sentence.py
# 計算内容：
# - 要約と推薦の平均文字数
# - テキストの多様性のためのDistinct-1/2スコア
# - テキスト類似性のためのBLEUとROUGEスコア
# - 統計的有意性検定
```

## 🔬 実験設定と評価

### 比較手法
- **ベースライン1 (コードではbaseline1 = 論文ではSumRec)**: DPO最適化なしの元のSumRec手法
- **ベースライン2 (コードではbaseline2 = 論文ではBaseline)**: アイテム推薦情報なしで対話要約のみを使用するシンプルベースライン
- **提案手法**: 要約と推薦の両方にDPOを使用した完全実装

**注意**: ソースコードでは、`baseline1`が論文の"SumRec"に対応し、`baseline2`が論文の"Baseline"に対応します。

### アブレーションスタディ
- **Rec-DPOなし (w/o Rec-DPO)**: アイテム推薦情報生成のDPOなしの提案手法
- **Sum-DPOなし (w/o Sum-DPO)**: 対話要約生成のDPOなしの提案手法

### 評価メトリクス

#### 定量的メトリクス
- **Hit Rate (HR)@k**: 正解アイテムが上位k推薦に現れるテストケースの割合
- **Mean Reciprocal Rank (MRR)@k**: 正解アイテムの逆順位の平均
- k = {1, 3, 5} で評価

#### 定性的メトリクス（クラウドワーカー評価）
- ベースラインと提案手法を比較する人間の好み評価
- 自然さと情報量に関するテキスト品質評価
- 統計的検証のために `metrics_sentence.py` を使用して分析

### ハイパーパラメータ最適化
- **ツール**: 自動ハイパーパラメータ探索のためのOptuna
- **最適化パラメータ**:
  - 学習率: [1e-7 から 5e-5]
  - バッチサイズ: [1, 2, 4, 8]
  - DPO βパラメータ: [0.01 から 0.5]
- **最適化試行**: モデルごとに5試行
- **選択基準**: 最高の検証性能

## 📈 実験結果

### 主要な結果
- **Tabidachiコーパス**: 既存手法と比較してすべてのメトリクス（HR@1,3,5、MRR@1,3,5）で優れた性能
- **ChatRecコーパス**: MRRで一貫して最高性能を達成

### 重要な発見
1. **対話要約生成のDPO訓練が特に重要**: アブレーションスタディで確認
2. **生成テキストの質的改善**: 推薦関連情報に焦点を当てたより詳細なテキスト生成
3. **上位ランクの改善**: 特にHR@1、MRR@1で顕著な性能改善

## 📁 ファイル構造

```
├── src/
│   ├── Tabidachi/                       # Tabidachi用メイン実装
│   │   ├── data_preprocessing.py        # 生データ前処理
│   │   ├── create_dataset_1.py          # 基本推薦データセット作成
│   │   ├── create_dataset_2.py          # スコア予測器データセット作成
│   │   ├── create_dataset_3.py          # 対話要約生成用DPOデータセット
│   │   ├── create_dataset_4.py          # アイテム推薦情報用DPOデータセット
│   │   ├── train_deberta.py             # DeBERTaスコア予測器訓練
│   │   ├── inference_deberta.py         # DeBERTa推論ユーティリティ
│   │   ├── dpo_summary_llm.py           # 対話要約生成モデル用DPO訓練（モデル1を作成）
│   │   ├── dpo_summary_llm_more.py      # 対話要約生成モデル用追加DPO訓練（モデル2-5を作成）
│   │   ├── dpo_summary_llm_cloudworker.py # クラウドワーカー評価用DPO訓練
│   │   ├── dpo_recommendation_llm.py    # アイテム推薦情報生成モデル用DPO訓練（モデル1を作成）
│   │   ├── dpo_recommendation_llm_more.py # アイテム推薦情報生成モデル用追加DPO訓練（モデル2-5を作成）
│   │   ├── create_recommend_data_proposal.py   # 提案手法推薦
│   │   ├── create_recommend_data_baseline1.py  # baseline1 = SumRec（論文）
│   │   ├── create_recommend_data_baseline2.py  # baseline2 = Baseline（論文）
│   │   ├── create_recommend_data_ablation1.py  # アブレーション：Rec-DPOなし
│   │   ├── create_recommend_data_ablation2.py  # アブレーション：Sum-DPOなし
│   │   ├── create_cloudworker_dataset.py       # クラウドワーカー評価データセット
│   │   ├── evaluate_from_recommend_data.py     # HR@k、MRR@k評価
│   │   ├── oss_llm.py                   # ベースLLMラッパー
│   │   ├── rec_model.py                 # アイテム推薦情報生成モデルラッパー
│   │   ├── summary_model.py             # 対話要約生成モデルラッパー
│   │   └── [ユーティリティスクリプト]    # create_csv*.py, count_*.py など
│   └── ChatRec/                         # ChatRec実装
│       ├── data_preprocessing.py        # 生データ前処理
│       ├── create_dataset_1.py          # 基本推薦データセット
│       ├── create_dataset_2.py          # スコア予測器データセット
│       ├── create_dataset_3.py          # 対話要約生成用DPOデータセット
│       ├── create_dataset_4.py          # アイテム推薦情報用DPOデータセット
│       ├── train_deberta.py             # DeBERTa訓練
│       ├── inference_deberta.py         # DeBERTa推論
│       ├── dpo_summary_llm.py           # 対話要約生成モデル用DPO訓練（モデル1を作成）
│       ├── dpo_summary_llm_more.py      # 対話要約生成モデル用追加DPO訓練（モデル2-5を作成）
│       ├── dpo_recommendation_llm.py    # アイテム推薦情報生成モデル用DPO訓練（モデル1を作成）
│       ├── dpo_recommendation_llm_more.py # アイテム推薦情報生成モデル用追加DPO訓練（モデル2-5を作成）
│       ├── create_recommend_data_proposal.py   # 提案手法
│       ├── create_recommend_data_baseline1.py  # baseline1 = SumRec（論文）
│       ├── create_recommend_data_baseline2.py  # baseline2 = Baseline（論文）
│       ├── evaluate_from_recommend_data.py     # 評価
│       ├── oss_llm.py                   # ベースLLMラッパー
│       ├── rec_model.py                 # アイテム推薦情報生成モデル
│       ├── summary_model.py             # 対話要約生成モデル
│       └── [ユーティリティスクリプト]    # count_*.py, logger.py
├── data/
│   ├── Tabidachi/                       # Tabidachiデータセット
│   │   ├── annotation_data/             # ダウンロードした元データ
│   │   ├── processed_data/              # 前処理済みデータ
│   │   ├── datasets_1/                  # 推薦データセット
│   │   ├── datasets_2/                  # スコア予測器訓練データセット
│   │   ├── datasets_3/                  # 対話要約生成用DPOデータセット
│   │   ├── datasets_4/                  # アイテム推薦情報用DPOデータセット
│   │   ├── recommend_data_*/            # 生成された推薦結果
│   │   └── cloudworker-dataset*/        # クラウドワーカー評価データ
│   └── ChatRec/                         # ChatRecデータセット
│       ├── chat_and_rec/                # ダウンロードした元データ
│       ├── processed_data/              # 前処理済みデータ
│       ├── datasets_1/                  # 推薦データセット
│       ├── datasets_2/                  # スコア予測器訓練データセット
│       ├── datasets_3/                  # 対話要約生成用DPOデータセット
│       ├── datasets_4/                  # アイテム推薦情報用DPOデータセット
│       └── recommend_data_*/            # 生成された推薦結果
├── images/                              # ドキュメント画像
│   ├── proposal_flow.pdf               # 手法フロー図（PDF）
│   └── proposal_flow.png               # 手法フロー図（PNG）
├── metrics_sentence.py                  # テキスト品質分析ツール
├── cloud_worker_tabidachi_datasets.xlsx # クラウドワーカー評価結果
├── requirements.txt                      # Python依存関係
├── README.md                            # 英語版README
└── README_JP.md                         # このファイル
```

## 📝 主要スクリプトの説明

### データ前処理スクリプト
- **`data_preprocessing.py`**: 生のアノテーションデータを実験用の構造化形式に変換
- **`create_dataset_1.py`**: 対話-候補ペアを持つ基本推薦データセットを作成
- **`create_dataset_2.py`**: スコア予測器（DeBERTa）用の訓練データを生成
- **`create_dataset_3.py`**: 対話要約生成DPO訓練用のプリファレンスペアを作成
- **`create_dataset_4.py`**: アイテム推薦情報DPO訓練用のプリファレンスペアを作成

### モデル訓練スクリプト
- **`train_deberta.py`**: 推薦スコアリング用のDeBERTaベースのスコア予測器を訓練
  - コマンドライン引数 `--method [proposal&baseline1|baseline2]` をサポート
  - METHOD_FLAGを自動設定：proposal&baseline1の場合True、baseline2の場合False
  - メソッド固有の出力ディレクトリを作成
- **`dpo_summary_llm.py`**: Optunaハイパーパラメータ最適化を使用した対話要約生成モデルのDPO訓練（モデル1を作成）
- **`dpo_summary_llm_more.py`**: 対話要約生成モデルの追加DPO訓練（モデル2-5を作成）
- **`dpo_summary_llm_cloudworker.py`**: 異なるデータ分割でのクラウドワーカー評価用特別訓練
- **`dpo_recommendation_llm.py`**: Optunaを使用したアイテム推薦情報生成モデルのDPO訓練（モデル1を作成）
- **`dpo_recommendation_llm_more.py`**: アイテム推薦情報生成モデルの追加DPO訓練（モデル2-5を作成）

### 推薦生成スクリプト
- **`create_recommend_data_proposal.py`**: 提案されたDPO訓練済みモデルを使用して推薦を生成
- **`create_recommend_data_baseline1.py`**: baseline1 = 論文のSumRec実装（DPOなし）
- **`create_recommend_data_baseline2.py`**: baseline2 = 論文のBaseline（アイテム推薦情報なしのシンプル手法）
- **`create_recommend_data_ablation1.py`**: 推薦DPOなしのアブレーションスタディ（Rec-DPOなし）
- **`create_recommend_data_ablation2.py`**: 要約DPOなしのアブレーションスタディ（Sum-DPOなし）
- **`create_cloudworker_dataset.py`**: 人間評価用の専用データセットを作成

### 評価スクリプト
- **`evaluate_from_recommend_data.py`**: すべての手法のHR@kとMRR@kメトリクスを計算
  - メソッド選択用のコマンドライン引数 `--method` をサポート
  - Tabidachi用：proposal、baseline1、baseline2、ablation1、ablation2
  - ChatRec用：proposal、baseline1、baseline2
  - proposal/ablationメソッドの場合、データセット1-5を自動的に読み込み
- **`metrics_sentence.py`**: 複数のメトリクスでテキスト品質を分析：
  - 要約と推薦の平均文字数
  - 多様性測定のためのDistinct-1/Distinct-2スコア
  - テキスト類似性のためのBLEUとROUGEスコア
  - `cloud_worker_tabidachi_datasets.xlsx` からクラウドワーカー評価結果を処理

### モデルラッパークラス
- **`oss_llm.py`**: Llama-3.1-Swallowモデル用ベースラッパー
- **`rec_model.py`**: アイテム推薦情報生成モデル用ラッパー
- **`summary_model.py`**: 対話要約生成モデル用ラッパー
- **`inference_deberta.py`**: DeBERTaスコア予測用ユーティリティ

## 🔄 完全な実験再現ガイド

### クイックスタート：論文結果の再現

#### Tabidachi実験の場合
```bash
# 1. 環境準備
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. データセットのダウンロードと配置
# https://www.nii.ac.jp/dsc/idr/rdata/Tabidachi/ からダウンロード
# data/Tabidachi/annotation_data/ に配置

# 3. 完全パイプラインの実行
cd src/Tabidachi
bash ../../scripts/run_tabidachi_experiments.sh  # スクリプトが存在する場合
# または手動実行：
python data_preprocessing.py
python create_dataset_1.py && python create_dataset_2.py
python create_dataset_3.py && python create_dataset_4.py
python train_deberta.py --method proposal&baseline1  # またはbaseline2
python dpo_summary_llm.py          # モデル1を作成
python dpo_summary_llm_more.py     # モデル2-5を作成
python dpo_recommendation_llm.py       # モデル1を作成
python dpo_recommendation_llm_more.py  # モデル2-5を作成
python create_recommend_data_proposal.py
python create_recommend_data_baseline1.py
python create_recommend_data_baseline2.py
python evaluate_from_recommend_data.py --method proposal  # 提案手法を評価
```

#### ChatRec実験の場合
```bash
# 1. データセットのダウンロードと配置
# https://github.com/Ryutaro-A/SumRec からクローン
# データを data/ChatRec/chat_and_rec/ に配置

# 2. 完全パイプラインの実行
cd src/ChatRec
python data_preprocessing.py
python create_dataset_1.py && python create_dataset_2.py
python create_dataset_3.py && python create_dataset_4.py
python train_deberta.py --method proposal&baseline1  # またはbaseline2
python dpo_summary_llm.py          # モデル1を作成
python dpo_summary_llm_more.py     # モデル2-5を作成
python dpo_recommendation_llm.py       # モデル1を作成
python dpo_recommendation_llm_more.py  # モデル2-5を作成
python create_recommend_data_proposal.py
python create_recommend_data_baseline1.py
python create_recommend_data_baseline2.py
python evaluate_from_recommend_data.py --method proposal  # 提案手法を評価
```

### モデル出力場所

#### Tabidachiモデル
- DeBERTa: `src/Tabidachi/deberta_best_model_proposal&baseline1/` または `deberta_best_model_baseline2/`
- DPO要約: `src/Tabidachi/dpo-summary-results_[1-5]/`
- DPO推薦: `src/Tabidachi/dpo-recommendation-results_[1-5]/`
- クラウドワーカーモデル: `src/Tabidachi/dpo-summary-results_cloudworker/`

#### ChatRecモデル
- DeBERTa: `src/ChatRec/ChatRec_deberta_best_model_proposal&baseline1/` または `ChatRec_deberta_best_model_baseline2/`
- DPO要約: `src/ChatRec/dpo-summary-results_[1-5]/`
- DPO推薦: `src/ChatRec/dpo-recommendation-results_[1-5]/`

### 結果出力場所
- Tabidachi推薦: `data/Tabidachi/recommend_data_[method]/`
- ChatRec推薦: `data/ChatRec/recommend_data_[method]/`
- 評価結果: コンソール出力とwandbログ

## 💻 実行時の考慮事項

### 使用ハードウェア
- **GPU**: Nvidia A100 80GB × 4
- **訓練時間**（4 × A100 80GBの場合）:
  - Llama-3.1-Swallow-8B DPO: 1エポックあたり約24時間（単一エポックで訓練）
  - DeBERTa: 1エポックあたり約4時間（1, 4, 10エポックでハイパーパラメータチューニング）

### 論文引用
**追加予定**

---

**注**: この実装は研究目的で作成されています。商用利用の場合は、各モデルとデータセットのライセンスを確認してください。