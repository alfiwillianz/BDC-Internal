# BDC-Internal: IELTS Essay Automated Scoring System

**Satria Data BDC 2025 Internal Selection Submission**

## 🎯 Project Overview

This project implements a sophisticated automated essay scoring system for IELTS (International English Language Testing System) essays using advanced machine learning techniques. Our solution predicts scores across four key assessment dimensions using deep text embeddings and comprehensive linguistic feature engineering.

## 📊 Dataset & Problem Statement

- **Task**: Multi-target regression for automated essay scoring
- **Dataset**: IELTS essays with prompts and human-scored rubrics
- **Training Data**: 9,912 essays with scored rubrics
- **Test Data**: 473 essays for prediction
- **Target Variables** (Score range 0-10):
  - `task_achievement`: How well the essay addresses the task requirements
  - `coherence_and_cohesion`: Logical organization and flow of ideas  
  - `lexical_resource`: Vocabulary range, accuracy, and appropriateness
  - `grammatical_range`: Grammar accuracy, complexity, and variety

## 🏗️ Architecture & Methodology

### 1. Text Preprocessing & Embedding Generation
- **Primary Embedding Model**: Alibaba-NLP/gte-Qwen2-7B-instruct
- **Feature Embedding Model**: BAAI/bge-large-en-v1.5
- **Optimization**: A100 GPU-optimized batch processing with FP16 precision
- **Text Cleaning**: Comprehensive normalization and preprocessing pipeline

### 2. Feature Engineering (54+ Features)

#### **Task Achievement Features** (15 features)
- Prompt-essay semantic similarity using transformer embeddings
- Word overlap and keyword coverage analysis
- Topic relevance and completeness metrics
- Named entity overlap with prompts

#### **Coherence & Cohesion Features** (18 features)
- Discourse marker detection and counting
- Sentence length statistics and variation
- Paragraph organization analysis
- Syntactic complexity measures

#### **Lexical Resource Features** (12 features)
- Advanced vocabulary diversity (TTR, Hapax/Dis Legomena, Yule's K, Honore's R)
- Readability scores (Flesch Reading Ease, Flesch-Kincaid, Dale-Chall)
- Word sophistication and syllable complexity
- Spelling error detection and frequency

#### **Grammatical Range Features** (9+ features)
- Grammar error detection using LanguageTool
- Part-of-speech distribution analysis
- Sentence structure variety and complexity
- Punctuation usage patterns

### 3. Model Architecture
- **Primary Model**: CatBoost with `MultiRMSEWithMissingValues` loss function
- **Dimensionality Reduction**: PCA (256 components) for embedding compression
- **Hyperparameter Optimization**: Optuna-based tuning with anchor submissions
- **Ensemble Strategy**: Multi-seed bagging for robust predictions

## 📁 Project Structure

```
BDC-Internal/
├── 00 Overview.ipynb              # Project introduction and objectives
├── 01 EDA.ipynb                   # Exploratory Data Analysis
├── 02 Feature Engineering.ipynb   # Comprehensive feature extraction
├── 03 Modelling.ipynb             # CatBoost model training
├── 04 Hyperparameter Tuning.ipynb # Optuna optimization
├── 05 PCA vs SVD Ensemble Pipeline.ipynb # Advanced ensemble methods
├── Anchors/                       # Submission anchors for optimization
├── df_train.csv                   # Training dataset
├── df_test.csv                    # Test dataset
└── README.md                      # This file
```

## 🚀 Key Technical Innovations

### 1. **Multi-Modal Feature Engineering**
- **Rule-based Features**: Fast, interpretable linguistic metrics
- **Deep Learning Features**: Transformer-based semantic embeddings
- **Domain-specific Features**: Expert knowledge about essay scoring

### 2. **Advanced Model Training**
- **Missing Value Handling**: CatBoost's native support for missing targets
- **GPU Optimization**: A100-optimized training with memory management
- **Anchor-based Optimization**: Using previous submissions as optimization targets

### 3. **Ensemble Methodology**
- **Multi-seed Training**: 5 different random seeds for robust predictions
- **Dimensionality Reduction Comparison**: PCA vs SVD analysis
- **Weighted Ensemble**: Performance-based prediction combination

## 📈 Results & Performance

### Model Performance
- **Ensemble Benefits**: Improved robustness through multi-seed training
- **Feature Importance**: Semantic similarity and discourse markers as top predictors

### Key Findings
1. **Semantic Similarity** is the strongest predictor across all dimensions
2. **Discourse Markers** strongly correlate with coherence scores
3. **Vocabulary Diversity** metrics effectively predict lexical resource scores
4. **Grammar Error Frequency** shows clear correlation with grammatical range

## 🔧 Usage Instructions

### Prerequisites
```bash
# Install required packages
pip install pandas numpy torch transformers sentence-transformers
pip install catboost optuna nltk spacy textstat language-tool-python
pip install scikit-learn matplotlib seaborn tqdm

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Running the Pipeline
1. **EDA**: `01 EDA.ipynb` - Explore the dataset characteristics
2. **Feature Engineering**: `02 Feature Engineering.ipynb` - Extract comprehensive features
3. **Modeling**: `03 Modelling.ipynb` - Train CatBoost models
4. **Hyperparameter Tuning**: `04 Hyperparameter Tuning.ipynb` - Optimize performance
5. **Ensemble**: `05 PCA vs SVD Ensemble Pipeline.ipynb` - Advanced ensemble methods

### Quick Start
```python
# Load and preprocess data
train = pd.read_csv('df_train.csv')
test = pd.read_csv('df_test.csv')

# Run feature engineering pipeline
train_features = engineer_complete_features(train)
test_features = engineer_complete_features(test)

# Train CatBoost model
model = CatBoostRegressor(loss_function="MultiRMSEWithMissingValues")
model.fit(train_features, targets)

# Generate predictions
predictions = model.predict(test_features)
```

## 🎯 Scoring Dimensions Analysis

### Task Achievement
- **Key Predictors**: Prompt similarity, topic coverage, essay completeness
- **Strategy**: Focus on semantic alignment with prompt requirements

### Coherence and Cohesion  
- **Key Predictors**: Discourse markers, sentence flow, paragraph organization
- **Strategy**: Emphasize logical structure and transition quality

### Lexical Resource
- **Key Predictors**: Vocabulary diversity, word sophistication, spelling accuracy
- **Strategy**: Balance vocabulary richness with appropriate usage

### Grammatical Range
- **Key Predictors**: Grammar accuracy, sentence complexity, POS variety
- **Strategy**: Reward both accuracy and structural sophistication

## 🔬 Advanced Features

### Memory-Optimized Processing
- FP16 precision for 50% memory reduction
- Chunked batch processing for large datasets
- Automatic GPU memory cleanup

### Robust Error Handling
- Graceful fallbacks for failed computations
- Timeout handling for external NLP tools
- Missing value imputation strategies

### Extensible Architecture
- Modular feature engineering functions
- Easy addition of new linguistic features
- Configurable model parameters

## 📊 Visualization & Insights

The notebooks include comprehensive visualizations:
- Feature importance analysis by scoring dimension
- Correlation matrices between features and targets
- Distribution analysis of linguistic metrics
- Performance comparison across ensemble methods

## 🏆 Competition Strategy

### Optimization Approach
1. **Anchor-based Tuning**: Use previous submissions as optimization targets
2. **Multi-seed Robustness**: Train with multiple random seeds
3. **Ensemble Diversity**: Combine PCA and SVD dimensionality reduction
4. **Feature Selection**: Focus on interpretable, domain-relevant features

### Key Success Factors
- Deep understanding of IELTS scoring criteria
- Comprehensive linguistic feature engineering
- Robust model architecture with missing value handling
- Performance-based ensemble weighting

## 🤝 Contributing

This project was developed for the Satria Data BDC 2025 Internal Selection. The approach combines academic research in automated essay scoring with practical machine learning engineering.

## 📚 References

- IELTS Assessment Criteria and Scoring Guidelines
- Academic research in automated essay scoring
- Advanced NLP techniques for text quality assessment
- Ensemble methods in educational data mining

---

**Team**: Team 5 - Satria Data BDC 2025 ITS Internal Selection  
**Approach**: Multi-target regression with comprehensive linguistic analysis  
**Innovation**: Anchor-based optimization with ensemble robustness