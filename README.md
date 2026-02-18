# NLP-DockerLint

A machine learning approach to Dockerfile linting using transformer models (CodeBERT and RoBERTa) for multi-class classification of Dockerfile issues.

## Project Structure

```
nlp-dockerlint/
├── configs/                  # Configuration files (hyperparameters, paths)
├── data/
│   ├── raw/                  # Raw data (dockerfiles, dockerfile_analysis.csv)
│   ├── processed/            # Train/val/test splits
│   └── tokenized/            # Tokenized datasets per model
├── models/                   # Trained model checkpoints
│   ├── stage1_codebert/
│   ├── stage1_roberta/
│   ├── stage2_codebert/
│   └── stage2_roberta/
├── notebooks/                # Jupyter notebooks for exploration
├── results/
│   ├── figures/              # ROC curves and visualizations
│   ├── metrics/              # Per-class performance CSVs
│   └── screenshots/          # Inference screenshots
├── src/
│   ├── data_collection/      # Dockerfile fetching from GitHub
│   ├── linting/              # Hadolint integration
│   │   ├── github/
│   │   └── henkel/
│   ├── models/               # Model training and evaluation
│   │   ├── codebert/
│   │   └── roberta/
│   ├── preprocessing/        # Data balancing and splitting
│   └── utils/                # Shared utilities
├── requirements.txt          # Python dependencies
└── README.md
```

## Pipeline Overview

```
[GitHub Search API]
        ↓
[Fetch Dockerfile URLs]
        ↓
[Download raw Dockerfile and save locally]
        ↓
[Run Hadolint on local file]
        ↓
[Parse issues → map to lines → assign labels]
        ↓
[Write labeled lines to CSV for training]
        ↓
[Train CodeBERT/RoBERTa models]
        ↓
[Evaluate and generate metrics]
```

## Two-Stage Classification

1. **Stage 1**: Binary classification (has issue / no issue)
2. **Stage 2**: Multi-class classification (specific issue type)

## Models

- **CodeBERT**: Pre-trained on code, optimized for programming language understanding
- **RoBERTa**: General-purpose language model baseline

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Data Tokenization

Pre-tokenized data is already available in `data/tokenized/` for both models:
- `data/tokenized/codebert/` - CodeBERT tokenized chunks
- `data/tokenized/roberta/` - RoBERTa tokenized chunks

You can skip this step and proceed directly to training if using the existing tokenized data.

**To re-tokenize from scratch:**

```bash
# CodeBERT Tokenization
python src/models/codebert/data-tokenization-stage1.py  # Stage 1 (binary)
python src/models/codebert/data-tokenization-stage2.py  # Stage 2 (multi-class)

# RoBERTa Tokenization
python src/models/roberta/data-tokenization-stage1.py   # Stage 1 (binary)
python src/models/roberta/data-tokenization-stage2.py   # Stage 2 (multi-class)
```

### 3. Training

#### CodeBERT

```bash
# Stage 1: Binary Classification (correct vs wrong)
python src/models/codebert/stage1-training.py

# Stage 2: Multi-class Classification (specific rule violated)
python src/models/codebert/stage2-training.py
```

#### RoBERTa

```bash
# Stage 1: Binary Classification (correct vs wrong)
python src/models/roberta/stage1-training.py

# Stage 2: Multi-class Classification (specific rule violated)
python src/models/roberta/stage2-training.py
```

**Training Notes:**
- Models automatically detect and use CUDA, MPS (Apple Silicon), or CPU
- Checkpoints are saved to `models/stage{1,2}_{codebert,roberta}/`
- Training can be resumed from the last checkpoint if interrupted

### 4. Performance Evaluation

```bash
# CodeBERT Performance
python src/models/codebert/stage1-performance.py           # Stage 1 metrics
python src/models/codebert/stage2-overall-performance.py   # Stage 2 overall metrics
python src/models/codebert/stage2-perclass-performance.py  # Stage 2 per-class metrics

# RoBERTa Performance
python src/models/roberta/stage1-performance.py            # Stage 1 metrics
python src/models/roberta/stage2-overall-performance.py    # Stage 2 overall metrics
python src/models/roberta/stage2-perclass-performance.py   # Stage 2 per-class metrics
```

### 5. Inference

Run inference on Dockerfile lines:

```bash
# CodeBERT Inference (two-stage pipeline)
python src/models/codebert/inference.py

# RoBERTa Inference (two-stage pipeline)
python src/models/roberta/inference.py
```

**Inference Pipeline:**
1. **Stage 1**: Classifies if the Dockerfile line is correct or wrong
2. **Stage 2**: If wrong, identifies the specific rule that was violated

## Results

See `results/metrics/` for detailed per-class performance metrics and `results/figures/` for ROC curves.
