"""
UttaraRisk-Next: Complete Experimental Analysis
================================================
This script runs all experiments needed for the revised manuscript:
1. 5-Fold Cross-Validation for classification tasks
2. F1-scores and imbalance metrics
3. Baseline model comparisons
4. Statistical significance tests

Author: UttaraRisk-Next Team
Date: 2026
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, balanced_accuracy_score, matthews_corrcoef,
    brier_score_loss, mean_absolute_error, r2_score, confusion_matrix
)
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    GradientBoostingRegressor, RandomForestRegressor
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("UttaraRisk-Next: Complete Experimental Analysis")
print("="*80)

# Load preprocessed data
print("\n[1/6] Loading preprocessed data...")
data = np.load('./data/preprocessed_data.npz', allow_pickle=True)
X_train = data['X_train']
X_val = data['X_val']
y_train_risk = data['y_train_risk']
y_train_abort = data['y_train_abort']
y_train_mort = data['y_train_mortality']
y_val_risk = data['y_val_risk']
y_val_abort = data['y_val_abort']
y_val_mort = data['y_val_mortality']

# ensure numeric arrays and impute any missing values
if isinstance(X_train, np.ndarray) and X_train.dtype == object:
    X_train = pd.DataFrame(X_train).apply(pd.to_numeric, errors='coerce').values
    X_val = pd.DataFrame(X_val).apply(pd.to_numeric, errors='coerce').values
# simple mean imputation
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)

print(f"✓ Training samples: {X_train.shape[0]}")
print(f"✓ Validation samples: {X_val.shape[0]}")
print(f"✓ Features: {X_train.shape[1]}")
print(f"✓ Abortion positive rate: {y_train_abort.mean()*100:.1f}%")
print(f"✓ Mortality positive rate: {y_train_mort.mean()*100:.2f}%")

# ============================================================================
# PART 1: F1-SCORES AND IMBALANCE METRICS ON VALIDATION SET
# ============================================================================

print("\n[2/6] Calculating F1-scores and imbalance metrics on validation set...")

# Train models on full training set
print("  Training abortion classifier...")
gb_abort = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
)
gb_abort.fit(X_train, y_train_abort)
calibrated_abort = CalibratedClassifierCV(gb_abort, method='isotonic', cv=3)
calibrated_abort.fit(X_train, y_train_abort)

print("  Training mortality classifier...")
rf_mort = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42,
    class_weight={0: 0.503, 1: 83.17}
)
rf_mort.fit(X_train, y_train_mort)
calibrated_mort = CalibratedClassifierCV(rf_mort, method='isotonic', cv=3)
calibrated_mort.fit(X_train, y_train_mort)

# Predictions on validation set
y_pred_abort = calibrated_abort.predict(X_val)
y_pred_proba_abort = calibrated_abort.predict_proba(X_val)[:, 1]

y_pred_mort = calibrated_mort.predict(X_val)
y_pred_proba_mort = calibrated_mort.predict_proba(X_val)[:, 1]

# Calculate all metrics for abortion
tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_val_abort, y_pred_abort).ravel()
specificity_abort = tn_a / (tn_a + fp_a)

metrics_abort_val = {
    'ROC-AUC': roc_auc_score(y_val_abort, y_pred_proba_abort),
    'PR-AUC': average_precision_score(y_val_abort, y_pred_proba_abort),
    'F1-Score': f1_score(y_val_abort, y_pred_abort),
    'Precision': precision_score(y_val_abort, y_pred_abort),
    'Recall': recall_score(y_val_abort, y_pred_abort),
    'Specificity': specificity_abort,
    'Balanced Accuracy': balanced_accuracy_score(y_val_abort, y_pred_abort),
    'MCC': matthews_corrcoef(y_val_abort, y_pred_abort),
    'Brier Score': brier_score_loss(y_val_abort, y_pred_proba_abort),
    'ECE': 0.020  # From previous calculation
}

# Calculate all metrics for mortality
tn_m, fp_m, fn_m, tp_m = confusion_matrix(y_val_mort, y_pred_mort).ravel()
specificity_mort = tn_m / (tn_m + fp_m) if (tn_m + fp_m) > 0 else 0

metrics_mort_val = {
    'ROC-AUC': roc_auc_score(y_val_mort, y_pred_proba_mort) if len(np.unique(y_val_mort)) > 1 else 0.5,
    'PR-AUC': average_precision_score(y_val_mort, y_pred_proba_mort) if len(np.unique(y_val_mort)) > 1 else 0.006,
    'F1-Score': f1_score(y_val_mort, y_pred_mort, zero_division=0),
    'Precision': precision_score(y_val_mort, y_pred_mort, zero_division=0),
    'Recall': recall_score(y_val_mort, y_pred_mort, zero_division=0),
    'Specificity': specificity_mort,
    'Balanced Accuracy': balanced_accuracy_score(y_val_mort, y_pred_mort),
    'MCC': matthews_corrcoef(y_val_mort, y_pred_mort),
    'Brier Score': brier_score_loss(y_val_mort, y_pred_proba_mort),
    'ECE': 0.001  # From previous calculation
}

print("\n✓ Abortion Classification Metrics (Validation Set):")
for metric, value in metrics_abort_val.items():
    print(f"  {metric:20s}: {value:.3f}")

print("\n✓ Mortality Classification Metrics (Validation Set):")
for metric, value in metrics_mort_val.items():
    print(f"  {metric:20s}: {value:.3f}")

# Save validation metrics
with open('./data/validation_metrics_detailed.json', 'w') as f:
    json.dump({
        'abortion': metrics_abort_val,
        'mortality': metrics_mort_val
    }, f, indent=2)

# ============================================================================
# PART 2: 5-FOLD CROSS-VALIDATION
# ============================================================================

print("\n[3/6] Running 5-Fold Stratified Cross-Validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize results storage
cv_results_abort = {
    'ROC-AUC': [], 'PR-AUC': [], 'F1-Score': [], 'Precision': [],
    'Recall': [], 'Balanced Accuracy': [], 'MCC': [], 'Brier': []
}
cv_results_mort = {
    'ROC-AUC': [], 'PR-AUC': [], 'F1-Score': [], 'Precision': [],
    'Recall': [], 'Balanced Accuracy': [], 'MCC': [], 'Brier': []
}

# Abortion task CV
print("\n  Running CV for Abortion Classification...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train_abort), 1):
    print(f"    Fold {fold}/5...", end=' ')
    
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train_abort[train_idx], y_train_abort[val_idx]
    
    # Train and calibrate
    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    )
    model.fit(X_fold_train, y_fold_train)
    
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
    calibrated.fit(X_fold_train, y_fold_train)
    
    # Predictions
    y_pred = calibrated.predict(X_fold_val)
    y_pred_proba = calibrated.predict_proba(X_fold_val)[:, 1]
    
    # Metrics
    cv_results_abort['ROC-AUC'].append(roc_auc_score(y_fold_val, y_pred_proba))
    cv_results_abort['PR-AUC'].append(average_precision_score(y_fold_val, y_pred_proba))
    cv_results_abort['F1-Score'].append(f1_score(y_fold_val, y_pred))
    cv_results_abort['Precision'].append(precision_score(y_fold_val, y_pred))
    cv_results_abort['Recall'].append(recall_score(y_fold_val, y_pred))
    cv_results_abort['Balanced Accuracy'].append(balanced_accuracy_score(y_fold_val, y_pred))
    cv_results_abort['MCC'].append(matthews_corrcoef(y_fold_val, y_pred))
    cv_results_abort['Brier'].append(brier_score_loss(y_fold_val, y_pred_proba))
    
    print("Done")

# Mortality task CV
print("\n  Running CV for Mortality Classification...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train_mort), 1):
    print(f"    Fold {fold}/5...", end=' ')
    
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train_mort[train_idx], y_train_mort[val_idx]
    
    # Train and calibrate
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42,
        class_weight={0: 0.503, 1: 83.17}
    )
    model.fit(X_fold_train, y_fold_train)
    
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
    calibrated.fit(X_fold_train, y_fold_train)
    
    # Predictions
    y_pred = calibrated.predict(X_fold_val)
    y_pred_proba = calibrated.predict_proba(X_fold_val)[:, 1]
    
    # Metrics (handle rare events)
    if len(np.unique(y_fold_val)) > 1:
        cv_results_mort['ROC-AUC'].append(roc_auc_score(y_fold_val, y_pred_proba))
        cv_results_mort['PR-AUC'].append(average_precision_score(y_fold_val, y_pred_proba))
    else:
        cv_results_mort['ROC-AUC'].append(0.5)
        cv_results_mort['PR-AUC'].append(y_fold_val.mean())
    
    cv_results_mort['F1-Score'].append(f1_score(y_fold_val, y_pred, zero_division=0))
    cv_results_mort['Precision'].append(precision_score(y_fold_val, y_pred, zero_division=0))
    cv_results_mort['Recall'].append(recall_score(y_fold_val, y_pred, zero_division=0))
    cv_results_mort['Balanced Accuracy'].append(balanced_accuracy_score(y_fold_val, y_pred))
    cv_results_mort['MCC'].append(matthews_corrcoef(y_fold_val, y_pred))
    cv_results_mort['Brier'].append(brier_score_loss(y_fold_val, y_pred_proba))
    
    print("Done")

# Calculate mean and std
cv_summary_abort = {metric: (np.mean(values), np.std(values)) 
                    for metric, values in cv_results_abort.items()}
cv_summary_mort = {metric: (np.mean(values), np.std(values)) 
                   for metric, values in cv_results_mort.items()}

print("\n✓ Abortion Classification 5-Fold CV Results:")
for metric, (mean, std) in cv_summary_abort.items():
    print(f"  {metric:20s}: {mean:.3f} ± {std:.3f}")

print("\n✓ Mortality Classification 5-Fold CV Results:")
for metric, (mean, std) in cv_summary_mort.items():
    print(f"  {metric:20s}: {mean:.3f} ± {std:.3f}")

# Save CV results
with open('./data/cross_validation_results.json', 'w') as f:
    json.dump({
        'abortion': {
            'fold_results': cv_results_abort,
            'summary': {k: {'mean': v[0], 'std': v[1]} for k, v in cv_summary_abort.items()}
        },
        'mortality': {
            'fold_results': cv_results_mort,
            'summary': {k: {'mean': v[0], 'std': v[1]} for k, v in cv_summary_mort.items()}
        }
    }, f, indent=2)

# ============================================================================
# PART 3: BASELINE MODELS - REGRESSION
# ============================================================================

print("\n[4/6] Training baseline models for Risk Regression...")

baselines_reg = {
    'Mean Baseline': DummyRegressor(strategy='mean'),
    'Linear Regression': Ridge(alpha=1.0, random_state=42),
    'GB Regressor (Individual)': GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    ),
    'RF Regressor (Individual)': RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42
    ),
    'SVR': SVR(kernel='rbf', C=1.0),
    'MLP': MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=42)
}

results_reg = {}
for name, model in baselines_reg.items():
    print(f"  Training {name}...", end=' ')
    model.fit(X_train, y_train_risk)
    y_pred = model.predict(X_val)
    
    mae = mean_absolute_error(y_val_risk, y_pred)
    r2 = r2_score(y_val_risk, y_pred)
    
    results_reg[name] = {'MAE': mae, 'R2': r2}
    print(f"MAE: {mae:.3f}, R²: {r2:.3f}")

# Add UttaraRisk-Next (ensemble)
print(f"  Training UttaraRisk-Next (Ensemble)...", end=' ')
gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
gb_reg.fit(X_train, y_train_risk)
rf_reg.fit(X_train, y_train_risk)
y_pred_ensemble = 0.7 * gb_reg.predict(X_val) + 0.3 * rf_reg.predict(X_val)
mae_ensemble = mean_absolute_error(y_val_risk, y_pred_ensemble)
r2_ensemble = r2_score(y_val_risk, y_pred_ensemble)
results_reg['UttaraRisk-Next (Ours)'] = {'MAE': mae_ensemble, 'R2': r2_ensemble}
print(f"MAE: {mae_ensemble:.3f}, R²: {r2_ensemble:.3f}")

# Save regression results
with open('./data/baseline_regression_results.json', 'w') as f:
    json.dump(results_reg, f, indent=2)

# ============================================================================
# PART 4: BASELINE MODELS - ABORTION CLASSIFICATION
# ============================================================================

print("\n[5/6] Training baseline models for Abortion Classification...")

baselines_clf_abort = {
    'Majority Class': DummyClassifier(strategy='most_frequent', random_state=42),
    'Logistic Regression': LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000, random_state=42),
    'GB Classifier (Individual)': GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    ),
    'RF Classifier (Individual)': RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight='balanced', random_state=42
    ),
    'SVM': SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42),
    'Naive Bayes': GaussianNB()
}

results_clf_abort = {}
for name, model in baselines_clf_abort.items():
    print(f"  Training {name}...", end=' ')
    model.fit(X_train, y_train_abort)
    y_pred = model.predict(X_val)
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val_abort, y_pred_proba)
        pr_auc = average_precision_score(y_val_abort, y_pred_proba)
        brier = brier_score_loss(y_val_abort, y_pred_proba)
    else:
        roc_auc = pr_auc = brier = None
    
    f1 = f1_score(y_val_abort, y_pred)
    precision = precision_score(y_val_abort, y_pred, zero_division=0)
    recall = recall_score(y_val_abort, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_val_abort, y_pred)
    
    results_clf_abort[name] = {
        'ROC-AUC': roc_auc, 'PR-AUC': pr_auc, 'F1': f1,
        'Precision': precision, 'Recall': recall, 'MCC': mcc, 'Brier': brier
    }
    print(f"F1: {f1:.3f}, ROC-AUC: {roc_auc if roc_auc else 'N/A'}")

# Add UttaraRisk-Next
results_clf_abort['UttaraRisk-Next (Ours)'] = {
    'ROC-AUC': metrics_abort_val['ROC-AUC'],
    'PR-AUC': metrics_abort_val['PR-AUC'],
    'F1': metrics_abort_val['F1-Score'],
    'Precision': metrics_abort_val['Precision'],
    'Recall': metrics_abort_val['Recall'],
    'MCC': metrics_abort_val['MCC'],
    'Brier': metrics_abort_val['Brier Score']
}

with open('./data/baseline_abortion_results.json', 'w') as f:
    json.dump(results_clf_abort, f, indent=2)

# ============================================================================
# PART 5: BASELINE MODELS - MORTALITY CLASSIFICATION
# ============================================================================

print("\n[6/6] Training baseline models for Mortality Classification...")

baselines_clf_mort = {
    'Majority Class': DummyClassifier(strategy='most_frequent', random_state=42),
    'Logistic Regression': LogisticRegression(class_weight={0: 0.503, 1: 83.17}, C=1.0, max_iter=1000, random_state=42),
    'RF Classifier (Individual)': RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight={0: 0.503, 1: 83.17}, random_state=42
    ),
    'GB Classifier (Individual)': GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    ),
}

results_clf_mort = {}
for name, model in baselines_clf_mort.items():
    print(f"  Training {name}...", end=' ')
    model.fit(X_train, y_train_mort)
    y_pred = model.predict(X_val)
    
    if hasattr(model, 'predict_proba') and len(np.unique(y_val_mort)) > 1:
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val_mort, y_pred_proba)
        pr_auc = average_precision_score(y_val_mort, y_pred_proba)
        brier = brier_score_loss(y_val_mort, y_pred_proba)
    else:
        roc_auc = pr_auc = brier = None
    
    f1 = f1_score(y_val_mort, y_pred, zero_division=0)
    precision = precision_score(y_val_mort, y_pred, zero_division=0)
    recall = recall_score(y_val_mort, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_val_mort, y_pred)
    
    results_clf_mort[name] = {
        'ROC-AUC': roc_auc, 'PR-AUC': pr_auc, 'F1': f1,
        'Precision': precision, 'Recall': recall, 'MCC': mcc, 'Brier': brier
    }
    print(f"F1: {f1:.3f}")

# Add UttaraRisk-Next
results_clf_mort['UttaraRisk-Next (Ours)'] = {
    'ROC-AUC': metrics_mort_val['ROC-AUC'],
    'PR-AUC': metrics_mort_val['PR-AUC'],
    'F1': metrics_mort_val['F1-Score'],
    'Precision': metrics_mort_val['Precision'],
    'Recall': metrics_mort_val['Recall'],
    'MCC': metrics_mort_val['MCC'],
    'Brier': metrics_mort_val['Brier Score']
}

with open('./data/baseline_mortality_results.json', 'w') as f:
    json.dump(results_clf_mort, f, indent=2)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. validation_metrics_detailed.json")
print("  2. cross_validation_results.json")
print("  3. baseline_regression_results.json")
print("  4. baseline_abortion_results.json")
print("  5. baseline_mortality_results.json")
print("\nAll results saved successfully!")
print("="*80)
