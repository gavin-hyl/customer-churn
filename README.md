# Customer Churn Prediction

A comprehensive machine learning project comparing multiple algorithms for predicting customer churn in the telecommunications industry.

## Overview

This project tackles the binary classification problem of predicting whether customers will discontinue their telecom services. Using a rich dataset of customer demographics, service usage, and account information, we implement and compare various machine learning approaches to achieve optimal prediction performance.

The project emphasizes practical machine learning workflows including data preprocessing, feature engineering, model selection, hyperparameter tuning, and ensemble methods for production-ready churn prediction.

## Dataset

### Customer Information
- **Size**: 5,344 training samples, 1,701 test samples
- **Features**: 20 customer attributes including demographics, services, and billing
- **Target**: Binary classification (Discontinued: Yes/No)
- **Class Balance**: Imbalanced dataset requiring careful handling

### Key Features
- **Demographics**: Gender, age, partner/dependent status
- **Services**: Phone, internet, streaming, security add-ons
- **Contract**: Month-to-month, one-year, two-year agreements
- **Billing**: Monthly charges, total charges, payment methods
- **Tenure**: Customer relationship duration

## Machine Learning Approaches

### 1. Random Forest
**Files**: `randomforest.ipynb`, `randomforest_kfold.ipynb`

Ensemble tree-based model with extensive hyperparameter optimization:

```python
model = RandomForestClassifier(
    n_estimators=1000, 
    criterion='entropy',
    min_samples_leaf=7, 
    max_features=1
)
```

**Key Results**:
- **Feature Importance**: Tenure, monthly charges, and contract type most predictive
- **Cross-Validation**: K-fold validation for robust performance estimation
- **Performance**: Strong baseline with good interpretability

### 2. Neural Networks
**File**: `neural.ipynb`

Deep learning approach for non-linear pattern recognition:

```python
# Multi-layer perceptron with dropout and regularization
model = Sequential([
    Dense(128, activation='relu', input_dim=n_features),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

**Features**:
- **Architecture**: Multi-layer perceptron with regularization
- **Optimization**: Adam optimizer with learning rate scheduling
- **Validation**: Early stopping to prevent overfitting

### 3. Boosting Algorithms

#### AdaBoost
**Files**: `adaboost.ipynb`, `adaboost_kfold.ipynb`, `adaboost_oversampling.ipynb`

Adaptive boosting with multiple optimization strategies:
- **K-Fold Validation**: Robust model evaluation
- **Oversampling**: Handling class imbalance with SMOTE
- **Best Score**: 0.85273 AUC achieved

#### XGBoost
**File**: `XGBoost_kfold.ipynb`

Gradient boosting with advanced regularization:
- **Cross-Validation**: K-fold for hyperparameter tuning
- **Feature Engineering**: Automated feature importance ranking
- **Scalability**: Efficient implementation for large datasets

#### Gradient Boosting
**File**: `gradient_boosting_0.84986.ipynb`

Scikit-learn gradient boosting implementation:
- **Performance**: 0.84986 AUC score
- **Interpretability**: Feature importance analysis
- **Robustness**: Strong generalization performance

### 4. AutoML Approach
**File**: `autogluon_test.ipynb`

Automated machine learning with AutoGluon:

```python
predictor = TabularPredictor(label='Discontinued')
predictor.fit(train_data, time_limit=600)
```

**Benefits**:
- **Automation**: Automated feature engineering and model selection
- **Ensemble**: Automatic model stacking and blending
- **Efficiency**: Minimal manual tuning required

### 5. Ensemble Methods
**File**: `ensemble.ipynb`

Model stacking and voting for improved performance:
- **Model Combination**: Weighted averaging of predictions
- **Diversity**: Leveraging different algorithm strengths
- **Robustness**: Reduced variance through ensemble averaging

## Data Processing Pipeline

### Feature Engineering
**File**: `utils.py`

Comprehensive preprocessing utilities:

```python
def numerize_csv(path, train=True, expand_classes=False, target_encode=False):
    # Convert categorical to numerical
    # Normalize continuous features
    # Handle missing values
    # Feature engineering
```

**Key Transformations**:
- **Categorical Encoding**: One-hot encoding vs. target encoding
- **Normalization**: Min-max scaling for numerical features
- **Feature Combination**: Related service features aggregated
- **Missing Value Handling**: Strategic imputation based on feature type

### Advanced Techniques
- **Target Encoding**: Categorical features encoded by target correlation
- **Feature Combination**: Related services combined into composite scores
- **Class Balancing**: Oversampling for imbalanced target distribution

## Performance Results

### Model Comparison
| Algorithm | Best AUC Score | Key Strengths |
|-----------|----------------|---------------|
| AdaBoost | 0.85273 | Robust to overfitting |
| Gradient Boosting | 0.84986 | Feature importance clarity |
| Random Forest | 0.84+ | Interpretability |
| XGBoost | 0.84+ | Scalability |
| Neural Networks | 0.83+ | Non-linear patterns |

### Feature Importance Rankings
1. **Tenure** - Customer relationship duration
2. **Monthly Charges** - Monthly service cost
3. **Contract Type** - Service agreement length
4. **Total Charges** - Cumulative billing amount
5. **Internet Service** - Type of internet plan

## Technical Implementation

### Dependencies
```python
import pandas as pd
import numpy as np
import scikit-learn
import xgboost
import autogluon
import torch  # for neural networks
```

### File Structure
```
cs155-project1/
├── README.md                     # This documentation
├── utils.py                      # Data processing utilities (148 lines)
├── randomforest.ipynb           # Random Forest implementation
├── randomforest_kfold.ipynb     # Random Forest with K-fold validation
├── neural.ipynb                 # Neural network approach
├── adaboost*.ipynb             # AdaBoost variants and optimization
├── XGBoost_kfold.ipynb         # XGBoost with cross-validation
├── gradient_boosting*.ipynb     # Gradient boosting experiments
├── autogluon_test.ipynb        # AutoML approach
├── ensemble.ipynb              # Model ensemble techniques
├── train.csv                   # Training dataset (727KB)
├── test.csv                    # Test dataset (224KB)
├── *_submission.csv            # Competition submission files
└── columns.pdf                 # Data dictionary
```

## Key Insights

### Data Analysis
- **Churn Drivers**: Short tenure and month-to-month contracts highly predictive
- **Service Patterns**: Fiber optic customers show higher churn rates
- **Billing Impact**: Higher monthly charges correlate with increased churn risk
- **Customer Segments**: Senior citizens and single customers more likely to churn

### Model Performance
- **Ensemble Benefits**: Combining models improves robustness
- **Feature Engineering**: Custom feature combinations boost performance
- **Class Imbalance**: Oversampling techniques provide marginal improvements
- **Cross-Validation**: Essential for reliable performance estimation

## Usage Examples

### Quick Model Training
```python
from utils import numerize_csv, combine_related_columns

# Load and preprocess data
data = combine_related_columns(numerize_csv('train.csv'))
X = data.drop('Discontinued', axis=1)
y = data['Discontinued']

# Train random forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000, criterion='entropy')
model.fit(X, y)
```

### Automated ML Pipeline
```python
from autogluon.tabular import TabularPredictor

# AutoML approach
predictor = TabularPredictor(label='Discontinued')
predictor.fit('train.csv', time_limit=600)
predictions = predictor.predict('test.csv')
```

### Feature Engineering
```python
# Custom feature combinations
df = combine_related_columns(df)  # Combine related service features
df = numerize_csv(df, expand_classes=True)  # One-hot encode categories
```

## Production Considerations

### Model Selection Criteria
- **Performance**: AUC score on validation set
- **Interpretability**: Feature importance for business insights
- **Scalability**: Training time and prediction latency
- **Robustness**: Performance across different customer segments

### Deployment Recommendations
- **AdaBoost**: Best overall performance with good interpretability
- **Random Forest**: Excellent baseline with fast inference
- **Ensemble**: Optimal performance for critical applications
- **AutoGluon**: Minimal maintenance overhead

## Business Applications

### Proactive Retention
- **Risk Scoring**: Identify high-risk customers for targeted intervention
- **Timing**: Optimal moments for retention campaigns
- **Personalization**: Tailored offers based on churn risk factors

### Strategic Insights
- **Contract Optimization**: Incentivize longer-term agreements
- **Service Bundling**: Reduce churn through strategic service combinations
- **Pricing Strategy**: Balance revenue and retention goals

---

*This project demonstrates end-to-end machine learning for customer churn prediction, comparing traditional algorithms with modern AutoML approaches to deliver actionable business insights.* 
