#!/usr/bin/env python3
"""
UttaraRisk-Next: Advanced Multi-Task Learning Model for Maternal Health Risk Prediction
Final implementation with proper data handling
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import json
import warnings
warnings.filterwarnings('ignore')

class ExpectedCalibrationError:
    """Expected Calibration Error computation"""
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
    
    def compute(self, y_true, y_prob):
        """Compute ECE"""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece

class UttaraRiskNextModel:
    """Advanced multi-task model for maternal health risk prediction"""
    
    def __init__(self, use_ensemble=True):
        self.use_ensemble = use_ensemble
        
        # Initialize models
        self.risk_model = None
        self.abort_model = None
        self.mortality_model = None
        
        # Calibration
        self.calibrated_abort_model = None
        self.calibrated_mortality_model = None
        
        # Feature importance
        self.feature_importance = {}
        
    def _get_class_weights(self, y):
        """Calculate class weights for imbalanced data"""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        class_weight = {}
        for i, (cls, count) in enumerate(zip(unique, counts)):
            class_weight[cls] = total / (len(unique) * count)
        return class_weight
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit the multi-task model"""
        print("=== TRAINING UTTARARISK-NEXT MODEL ===")
        
        # Convert to numpy arrays if pandas DataFrames
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if X_val is not None and hasattr(X_val, 'values'):
            X_val = X_val.values

        # If array has object dtype (mixed types or strings), coerce to numeric
        if isinstance(X_train, np.ndarray) and X_train.dtype == object:
            df_temp = pd.DataFrame(X_train)
            df_temp = df_temp.apply(pd.to_numeric, errors='coerce')
            X_train = df_temp.values
            if X_val is not None:
                df_temp2 = pd.DataFrame(X_val)
                df_temp2 = df_temp2.apply(pd.to_numeric, errors='coerce')
                X_val = df_temp2.values

        # Impute missing values using mean strategy if necessary
        # (GradientBoostingRegressor/Classifier don't accept NaN)
        # wrap NaN check in try/except in case dtype still unsupported
        nan_present = False
        try:
            nan_present = np.isnan(X_train).any()
        except TypeError:
            # fallback to pandas
            nan_present = pd.DataFrame(X_train).isna().any().any()

        if nan_present:
            print("Detected NaNs in X_train, applying SimpleImputer...")
            self.imputer = SimpleImputer(strategy='mean')
            X_train = self.imputer.fit_transform(X_train)
            if X_val is not None:
                X_val = self.imputer.transform(X_val)
        elif hasattr(self, 'imputer') and X_val is not None:
            # if imputer exists (already fitted) use it on validation
            X_val = self.imputer.transform(X_val)
            
        # Extract targets
        y_risk = y_train['risk']
        y_abort = y_train['abort']
        y_mortality = y_train['mortality']
        
        print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        
        # 1. Risk prediction (regression) - Ensemble approach
        print("1. Training risk prediction model (Ensemble)...")
        if self.use_ensemble:
            # Combine Gradient Boosting and Random Forest
            gb_risk = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                random_state=42
            )
            
            rf_risk = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=5,
                random_state=42
            )
            
            gb_risk.fit(X_train, y_risk)
            rf_risk.fit(X_train, y_risk)
            
            self.risk_model = {'gb': gb_risk, 'rf': rf_risk}
        else:
            self.risk_model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                random_state=42
            )
            self.risk_model.fit(X_train, y_risk)
        
        # 2. Abortion prediction (classification)
        print("2. Training abortion prediction model...")
        abort_class_weights = self._get_class_weights(y_abort)
        
        self.abort_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            random_state=42
        )
        
        # Apply class weights by sample weighting
        sample_weights_abort = np.array([abort_class_weights[y] for y in y_abort])
        self.abort_model.fit(X_train, y_abort, sample_weight=sample_weights_abort)
        
        # 3. Mortality prediction (classification)
        print("3. Training mortality prediction model...")
        mortality_class_weights = self._get_class_weights(y_mortality)
        
        # Use Random Forest for better handling of rare events
        self.mortality_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=mortality_class_weights,
            random_state=42
        )
        
        self.mortality_model.fit(X_train, y_mortality)
        
        # 4. Calibration
        print("4. Applying probability calibration...")
        self.calibrated_abort_model = CalibratedClassifierCV(
            self.abort_model, method='isotonic', cv=3
        )
        self.calibrated_abort_model.fit(X_train, y_abort)
        
        self.calibrated_mortality_model = CalibratedClassifierCV(
            self.mortality_model, method='isotonic', cv=3
        )
        self.calibrated_mortality_model.fit(X_train, y_mortality)
        
        # Store feature importance
        if self.use_ensemble:
            self.feature_importance['risk'] = (self.risk_model['gb'].feature_importances_ + 
                                             self.risk_model['rf'].feature_importances_) / 2
        else:
            self.feature_importance['risk'] = self.risk_model.feature_importances_
        self.feature_importance['abort'] = self.abort_model.feature_importances_
        self.feature_importance['mortality'] = self.mortality_model.feature_importances_
        
        print("✓ Model training completed!")
        
    def predict(self, X):
        """Make predictions for all tasks"""
        # Convert to numpy array if pandas DataFrame
        if hasattr(X, 'values'):
            X = X.values

        # Apply imputer if it was fitted during training
        if hasattr(self, 'imputer'):
            X = self.imputer.transform(X)
            
        # Risk prediction with ensemble
        if self.use_ensemble and isinstance(self.risk_model, dict):
            gb_pred = self.risk_model['gb'].predict(X)
            rf_pred = self.risk_model['rf'].predict(X)
            risk_pred = (gb_pred + rf_pred) / 2  # Simple ensemble
        else:
            risk_pred = self.risk_model.predict(X)
        
        # For quantile intervals (using prediction uncertainty estimation)
        risk_std = np.std(risk_pred) * 1.645  # Approximate 90% interval
        risk_lo_90 = np.clip(risk_pred - risk_std, 0, 100)
        risk_hi_90 = np.clip(risk_pred + risk_std, 0, 100)
        
        # Abortion prediction (calibrated)
        abort_prob = self.calibrated_abort_model.predict_proba(X)[:, 1]
        
        # Mortality prediction (calibrated)
        mortality_prob = 1 - self.calibrated_mortality_model.predict_proba(X)[:, 1]  # Prob of death
        
        return {
            'risk_to_woman_pct': risk_pred,
            'risk_lo_90': risk_lo_90,
            'risk_hi_90': risk_hi_90,
            'aborted_prob': abort_prob,
            'mother_alive_dead_prob': 1 - mortality_prob  # Prob of survival
        }
    
    def evaluate(self, X_val, y_val):
        """Comprehensive model evaluation"""
        print("\n=== MODEL EVALUATION ===")
        
        predictions = self.predict(X_val)
        
        # Regression metrics (risk prediction)
        risk_mae = mean_absolute_error(y_val['risk'], predictions['risk_to_woman_pct'])
        risk_r2 = r2_score(y_val['risk'], predictions['risk_to_woman_pct'])
        
        # Coverage of 90% interval
        in_interval = ((y_val['risk'] >= predictions['risk_lo_90']) & 
                      (y_val['risk'] <= predictions['risk_hi_90']))
        interval_coverage = in_interval.mean()
        
        # Classification metrics
        abort_auc = roc_auc_score(y_val['abort'], predictions['aborted_prob'])
        abort_prauc = average_precision_score(y_val['abort'], predictions['aborted_prob'])
        abort_brier = brier_score_loss(y_val['abort'], predictions['aborted_prob'])
        
        # Mortality metrics (handle rare events)
        if len(np.unique(y_val['mortality'])) > 1:
            mortality_auc = roc_auc_score(y_val['mortality'], 1 - predictions['mother_alive_dead_prob'])
            mortality_prauc = average_precision_score(1 - y_val['mortality'], 1 - predictions['mother_alive_dead_prob'])
            mortality_brier = brier_score_loss(1 - y_val['mortality'], 1 - predictions['mother_alive_dead_prob'])
        else:
            mortality_auc = mortality_prauc = mortality_brier = np.nan
        
        # Calibration
        ece_calculator = ExpectedCalibrationError()
        abort_ece = ece_calculator.compute(y_val['abort'], predictions['aborted_prob'])
        
        if not np.isnan(mortality_auc):
            mortality_ece = ece_calculator.compute(1 - y_val['mortality'], 1 - predictions['mother_alive_dead_prob'])
        else:
            mortality_ece = np.nan
        
        # Recall at 90% precision (for mortality)
        def recall_at_precision(y_true, y_prob, target_precision=0.9):
            if len(np.unique(y_true)) <= 1 or y_true.sum() == 0:
                return np.nan
            
            # Sort by probability (descending)
            sorted_indices = np.argsort(y_prob)[::-1]
            y_true_sorted = y_true[sorted_indices]
            
            # Find threshold for target precision
            for i in range(1, len(y_true_sorted) + 1):
                if y_true_sorted[:i].sum() == 0:
                    continue
                precision = y_true_sorted[:i].sum() / i
                if precision >= target_precision:
                    recall = y_true_sorted[:i].sum() / y_true.sum()
                    return recall
            return 0.0
        
        mortality_recall_90p = recall_at_precision(1 - y_val['mortality'], 1 - predictions['mother_alive_dead_prob'])
        
        metrics = {
            'regression': {
                'risk_mae': float(risk_mae),
                'risk_r2': float(risk_r2),
                'interval_coverage_90': float(interval_coverage)
            },
            'abortion_classification': {
                'roc_auc': float(abort_auc),
                'pr_auc': float(abort_prauc),
                'brier_score': float(abort_brier),
                'ece': float(abort_ece)
            },
            'mortality_classification': {
                'roc_auc': float(mortality_auc) if not np.isnan(mortality_auc) else None,
                'pr_auc': float(mortality_prauc) if not np.isnan(mortality_prauc) else None,
                'brier_score': float(mortality_brier) if not np.isnan(mortality_brier) else None,
                'ece': float(mortality_ece) if not np.isnan(mortality_ece) else None,
                'recall_at_90p_precision': float(mortality_recall_90p) if not np.isnan(mortality_recall_90p) else None
            }
        }
        
        # Print metrics
        print(f"Risk Prediction (Regression):")
        print(f"  MAE: {risk_mae:.3f}")
        print(f"  R²: {risk_r2:.3f}")
        print(f"  90% Interval Coverage: {interval_coverage:.3f}")
        
        print(f"\nAbortion Prediction (Classification):")
        print(f"  ROC-AUC: {abort_auc:.3f}")
        print(f"  PR-AUC: {abort_prauc:.3f}")
        print(f"  Brier Score: {abort_brier:.3f}")
        print(f"  ECE: {abort_ece:.3f}")
        
        print(f"\nMaternal Mortality Prediction (Classification):")
        if not np.isnan(mortality_auc):
            print(f"  ROC-AUC: {mortality_auc:.3f}")
            print(f"  PR-AUC: {mortality_prauc:.3f}")
            print(f"  Brier Score: {mortality_brier:.3f}")
            print(f"  ECE: {mortality_ece:.3f}")
            print(f"  Recall at 90% Precision: {mortality_recall_90p:.3f}")
        else:
            print("  Insufficient mortality events for evaluation")
        
        return metrics, predictions
    
    def get_feature_importance(self, feature_names):
        """Get feature importance rankings"""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'risk_importance': self.feature_importance['risk'],
            'abort_importance': self.feature_importance['abort'],
            'mortality_importance': self.feature_importance['mortality']
        })
        
        # Calculate average importance
        importance_df['avg_importance'] = (importance_df['risk_importance'] + 
                                         importance_df['abort_importance'] + 
                                         importance_df['mortality_importance']) / 3
        
        return importance_df.sort_values('avg_importance', ascending=False)

# Load preprocessed data and train model
if __name__ == "__main__":
    # Load data
    data = np.load('./data/preprocessed_data.npz', allow_pickle=True)
    X_train, X_val = data['X_train'], data['X_val']
    
    y_train = {
        'risk': data['y_train_risk'],
        'abort': data['y_train_abort'],
        'mortality': data['y_train_mortality']
    }
    
    y_val = {
        'risk': data['y_val_risk'],
        'abort': data['y_val_abort'], 
        'mortality': data['y_val_mortality']
    }
    
    # Initialize and train model
    model = UttaraRiskNextModel(use_ensemble=True)
    model.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    metrics, predictions = model.evaluate(X_val, y_val)
    
    # Save results
    with open('./data/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create predictions dataframe and save
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv('./data/predictions.csv', index=False)
    
    print("\n✓ Model training and evaluation completed!")
    print("✓ Saved metrics.json")
    print("✓ Saved predictions.csv")