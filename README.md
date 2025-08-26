# Refining Text Generation for Realistic Conversational Recommendation via Direct Preference Optimization

This repository contains the implementation code for the research paper "Refining Text Generation for Realistic Conversational Recommendation via Direct Preference Optimization". We propose a method to improve text generation in Conversational Recommender Systems (CRSs) using Direct Preference Optimization (DPO).

## 📖 Overview

![Proposal Method Flow](images/proposal_flow.png)
*[View high-resolution diagram (PDF)](images/proposal_flow.pdf)*

Traditional Conversational Recommender Systems (CRSs) face challenges such as hasty recommendations in short conversations and insufficient integration of implicit information. This research extends SumRec by applying DPO to optimize both dialogue summarization and item recommendation text generation models, achieving more realistic and natural conversational recommendation.

### Key Contributions
1. Extension of SumRec using DPO to propose a recommendation method suitable for realistic conversational recommendation datasets
2. Demonstration of superior recommendation performance through comparison with baseline methods and the original SumRec

## 🏗️ System Architecture

### Two-Stage Training Approach
- **Stage 1**: Pre-training of score predictor (DeBERTa)
- **Stage 2**: DPO training for dialogue summarization and recommendation text generation models

### Model Configuration
- **Base Model**: Llama-3.1-Swallow-8B-v0.1
- **Score Predictor**: DeBERTa-v3-japanese-large
- **Optimization Method**: Direct Preference Optimization (DPO)

## 📊 Datasets and Model Sources

### Datasets
- **Tabidachi Corpus**: https://www.nii.ac.jp/dsc/idr/rdata/Tabidachi/
  - Travel agent recommendation dialogues (realistic long conversations)
- **ChatRec Dataset**: https://github.com/Ryutaro-A/SumRec
  - Multi-category recommendation dialogues (for comparison)

### Pre-trained Models
- **Llama-3.1-Swallow-8B-v0.1**: https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-v0.1
- **DeBERTa-v3-japanese-large**: https://huggingface.co/globis-university/deberta-v3-japanese-large

## 🛠️ Environment Setup

### Requirements
- Python 3.10.12
- GPU: Nvidia A100 80GB × 4 (recommended)
- CUDA-compatible environment

### Setup Instructions

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
- PyTorch 2.4.1
- Transformers 4.46.2
- TRL (Transformer Reinforcement Learning) 0.12.1
- Optuna 4.1.0
- Datasets 3.1.0
- See `requirements.txt` for complete details

## 📂 Dataset Configuration

This project uses four types of dataset formats:

- **datasets_1**: Recommendation datasets (for dialogue summarization and item recommendation text generation)
- **datasets_2**: Score predictor training datasets
- **datasets_3**: DPO datasets for dialogue summarization models
- **datasets_4**: DPO datasets for item recommendation text generation models

### Data Splitting

#### Tabidachi Corpus
- **Splitting Method**: User ID-based
- **Data Composition**:
  - Training data: 20 adults, 7 elderly, 15 children
  - Validation data: 2 adults, 1 elderly, 2 children
  - Test data: 3 adults, 2 elderly, 3 children
- **User ID Ranges**:
  - Adults: 101-125 (25 users)
  - Elderly: 201-210 (10 users)
  - Children: 301-320 (20 users)
- **Scoring**: Actually recommended items = 1, others = 0

#### ChatRec Corpus
- **Splitting Method**: Category-based
- **Data Count by Category**:
  - except_for_travel: 223 items (train: 178, test: 33, validation: 12)
  - travel: 237 items (train: 189, test: 35, validation: 13)
  - no_restriction: 545 items (train: 436, test: 81, validation: 28)
- **Score Conversion**: 
  - Human-predicted scores (1-5 scale by 5 third-party workers) → Scores <3 converted to 0 (dislike), scores ≥3 converted to 1 (like)
  - This conversion unifies the scoring with Tabidachi corpus binary classification

#### Common Specifications
- **Ratio**: Train:Test:Validation = 8:1.5:0.5

## 🚀 Execution Workflow

### 1. Data Preprocessing
```bash
# Create datasets (execute in order)
python src/create_dataset_1.py
python src/create_dataset_2.py
python src/create_dataset_3.py
python src/create_dataset_4.py
```

### 2. Score Predictor Training
```bash
python src/train_deberta.py
```

### 3. DPO Training
```bash
# DPO training for dialogue summarization model
python src/dpo_summary_llm.py

# DPO training for recommendation text generation model
python src/dpo_recommendation_llm.py
```

### 4. Recommendation Data Generation and Evaluation
```bash
# Generate recommendation data using proposed method
python src/create_recommend_data_proposal.py

# Generate recommendation data using baseline methods
python src/create_recommend_data_baseline1.py
python src/create_recommend_data_baseline2.py

# Execute evaluation
python src/evaluate_from_recommend_data.py
```

## 🔬 Experimental Setup and Evaluation

### Comparison Methods
- **Baseline**: Uses only dialogue summaries, no recommendation text generation
- **SumRec**: Original method before DPO application
- **Proposed**: Proposed method with DPO applied

### Ablation Studies
- **w/o Rec-DPO**: Without DPO for recommendation text generation model
- **w/o Sum-DPO**: Without DPO for dialogue summarization model

### Evaluation Metrics
- **Hit Rate (HR)@k**: Proportion of cases where the correct answer is included in top-k items
- **Mean Reciprocal Rank (MRR)@k**: Average of reciprocal ranks of correct items
- Evaluated at k = 1, 3, 5

### Hyperparameter Optimization
- Automatic optimization using Optuna
- Optimization targets: learning rate, batch size, DPO β parameter

## 📈 Experimental Results

### Key Results
- **Tabidachi Corpus**: Superior performance across all metrics (HR@1,3,5, MRR@1,3,5) compared to existing methods
- **ChatRec Corpus**: Consistently achieved best performance in MRR

### Important Findings
1. **DPO training for dialogue summaries is particularly important**: Confirmed through ablation studies
2. **Qualitative improvement in generated text**: More detailed text generation focusing on recommendation-related information
3. **Improvement in top ranks**: Notable performance improvement especially in HR@1, MRR@1

## 📁 File Structure

```
├── src/                          # Main implementation (for Tabidachi)
│   ├── create_dataset_*.py       # Dataset creation scripts
│   ├── train_deberta.py          # Score predictor training
│   ├── dpo_summary_llm.py        # Dialogue summarization DPO training
│   ├── dpo_recommendation_llm.py # Recommendation text DPO training
│   ├── create_recommend_data_*.py # Recommendation data generation
│   ├── evaluate_from_recommend_data.py # Evaluation script
│   ├── rec_model.py              # Recommendation model
│   └── summary_model.py          # Summarization model
├── src-chatrec/                  # ChatRec implementation
├── images/                       # Images for README
│   ├── proposal_flow.pdf         # Proposal method flow diagram (PDF)
│   └── proposal_flow.png         # Proposal method flow diagram (PNG)
├── metrics_sentence.py           # Text quality evaluation
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

### Key Script Descriptions

#### Data Preprocessing
- `create_dataset_1.py`: Basic recommendation dataset creation
- `create_dataset_2.py`: Score predictor dataset creation
- `create_dataset_3.py`: Preference data creation for dialogue summarization DPO
- `create_dataset_4.py`: Preference data creation for recommendation text DPO

#### Model Training
- `train_deberta.py`: Training of DeBERTa-based score predictor
- `dpo_summary_llm.py`: DPO training for dialogue summarization model using Optuna
- `dpo_recommendation_llm.py`: DPO training for recommendation text generation model using Optuna

#### Recommendation and Evaluation
- `create_recommend_data_proposal.py`: Recommendation execution using proposed method
- `create_recommend_data_baseline*.py`: Recommendation execution using baseline methods
- `evaluate_from_recommend_data.py`: Execution of HR@k, MRR@k evaluation

## 💻 Runtime Considerations

### GPU and Memory Requirements
- Recommended: Nvidia A100 80GB × 4
- Llama-3.1-Swallow-8B training: Approximately 24 hours (using 4 GPUs)
- DeBERTa training: Approximately 4 hours (using 4 GPUs)

### Paper Citation
**To be added**

---

**Note**: This implementation is created for research purposes. For commercial use, please verify the licenses of each model and dataset.