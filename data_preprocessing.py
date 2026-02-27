#!/usr/bin/env python3
"""
Advanced data preprocessing pipeline for UttaraRisk-Next model
Includes clinical feature engineering, missing value imputation, and fairness-aware splits
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class UttarakhandDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputers = {}
        self.feature_names = []
        
    def load_data(self, filepath):
        """Load and initial data preparation"""
        df = pd.read_csv(filepath)
        print(f"Loaded data: {df.shape}")
        return df
    
    def create_clinical_features(self, df):
        """Create clinically meaningful feature buckets"""
        df = df.copy()
        
        # Hemoglobin categories (WHO criteria for pregnant women)
        def categorize_hemoglobin(hb):
            if pd.isna(hb):
                return 'Unknown'
            elif hb < 7:
                return 'Severe_Anemia'
            elif hb < 10:
                return 'Moderate_Anemia'  
            elif hb < 11:
                return 'Mild_Anemia'
            else:
                return 'Normal'
        
        df['hb_category'] = df['hemoglobin'].apply(categorize_hemoglobin)
        
        # Blood pressure categories (American Heart Association)
        def categorize_bp(systolic, diastolic):
            if pd.isna(systolic) or pd.isna(diastolic):
                return 'Unknown'
            elif systolic >= 140 or diastolic >= 90:
                return 'Hypertensive'
            elif systolic >= 130 or diastolic >= 80:
                return 'Stage1_HTN'
            elif systolic >= 120:
                return 'Elevated'
            else:
                return 'Normal'
        
        df['bp_category'] = df.apply(lambda x: categorize_bp(x['systolic_bp'], x['diastolic_bp']), axis=1)
        
        # BMI categories (WHO)
        def categorize_bmi(bmi):
            if pd.isna(bmi):
                return 'Unknown'
            elif bmi < 18.5:
                return 'Underweight'
            elif bmi < 25:
                return 'Normal'
            elif bmi < 30:
                return 'Overweight'
            else:
                return 'Obese'
        
        df['bmi_category'] = df['bmi'].apply(categorize_bmi)
        
        # Age risk categories
        def categorize_age_risk(age):
            if pd.isna(age):
                return 'Unknown'
            elif age < 20:
                return 'Teen_Pregnancy'
            elif age > 35:
                return 'Advanced_Maternal_Age'
            else:
                return 'Optimal_Age'
        
        df['age_risk_category'] = df['age'].apply(categorize_age_risk)
        
        # Gestational age categories
        def categorize_gestational_age(ga):
            if pd.isna(ga):
                return 'Unknown'
            elif ga < 28:
                return 'Extremely_Preterm'
            elif ga < 32:
                return 'Very_Preterm'
            elif ga < 37:
                return 'Moderate_Preterm'
            else:
                return 'Term'
        
        df['ga_category'] = df['gestational_age'].apply(categorize_gestational_age)
        
        # ANC adequacy
        def categorize_anc(visits):
            if pd.isna(visits):
                return 'Unknown'
            elif visits < 4:
                return 'Inadequate'
            elif visits >= 8:
                return 'Adequate_Plus'
            else:
                return 'Adequate'
        
        df['anc_adequacy'] = df['anc_visits'].apply(categorize_anc)
        
        # Distance to healthcare access
        def categorize_distance(distance):
            if pd.isna(distance):
                return 'Unknown'
            elif distance <= 5:
                return 'Near'
            elif distance <= 15:
                return 'Moderate'
            else:
                return 'Far'
        
        df['healthcare_access'] = df['distance_to_hospital'].apply(categorize_distance)
        
        # Socioeconomic vulnerability index
        def create_vulnerability_score(row):
            score = 0
            if row.get('bpl_card') == 1:
                score += 2
            if row.get('education') in ['Illiterate', 'Primary']:
                score += 1
            if row.get('caste') in ['SC', 'ST']:
                score += 1
            if row.get('rural_urban') == 'Rural':
                score += 1
            return score
        
        df['vulnerability_score'] = df.apply(create_vulnerability_score, axis=1)
        
        # High-risk pregnancy indicators
        df['high_risk_pregnancy'] = (
            (df['age'] < 20) | (df['age'] > 35) |
            (df['hemoglobin'] < 8) |
            (df['systolic_bp'] > 140) | (df['diastolic_bp'] > 90) |
            (df['diabetes'] == 1) | (df['hypertension'] == 1) |
            (df['previous_cesarean'] == 1) |
            (df['bmi'] < 18.5) | (df['bmi'] > 30)
        ).astype(int)
        
        return df
    
    def handle_missing_values(self, df):
        """Advanced missing value imputation with missingness flags"""
        df = df.copy()
        
        # Create missingness flags for important variables
        missing_vars = ['hemoglobin', 'bmi', 'gestational_age', 'anc_visits', 
                       'systolic_bp', 'diastolic_bp', 'distance_to_hospital']
        
        for var in missing_vars:
            if var in df.columns:
                df[f'{var}_missing'] = df[var].isnull().astype(int)
        
        # Impute numerical variables
        numerical_vars = ['age', 'hemoglobin', 'bmi', 'gestational_age', 
                         'anc_visits', 'systolic_bp', 'diastolic_bp', 
                         'distance_to_hospital', 'gravida', 'para']
        
        for var in numerical_vars:
            if var in df.columns:
                if df[var].isnull().sum() > 0:
                    # Use median for skewed distributions, mean for normal
                    if df[var].skew() > 1:
                        fill_value = df[var].median()
                    else:
                        fill_value = df[var].mean()
                    df[var].fillna(fill_value, inplace=True)
        
        # Impute categorical variables
        categorical_vars = ['caste', 'education']
        for var in categorical_vars:
            if var in df.columns:
                mode_value = df[var].mode()[0] if len(df[var].mode()) > 0 else 'Unknown'
                df[var].fillna(mode_value, inplace=True)
        
        # Impute binary variables
        binary_vars = ['bpl_card', 'diabetes', 'hypertension', 'previous_cesarean']
        for var in binary_vars:
            if var in df.columns:
                mode_value = df[var].mode()[0] if len(df[var].mode()) > 0 else 0
                df[var].fillna(mode_value, inplace=True)
        
        return df
    
    def encode_categorical_variables(self, df):
        """One-hot encode categorical variables"""
        df = df.copy()
        
        # Variables to one-hot encode
        categorical_vars = ['district', 'rural_urban', 'caste', 'education',
                           'hb_category', 'bp_category', 'bmi_category', 
                           'age_risk_category', 'ga_category', 'anc_adequacy',
                           'healthcare_access']
        
        # One-hot encode
        for var in categorical_vars:
            if var in df.columns:
                dummies = pd.get_dummies(df[var], prefix=var, dummy_na=False)
                df = pd.concat([df, dummies], axis=1)
                df.drop(var, axis=1, inplace=True)
        
        return df
    
    def create_train_val_split(self, df):
        """Create stratified train-validation split with group consideration"""
        
        # Prepare target variables
        y_risk = df['risk_to_woman_pct'].values
        y_abort = df['aborted'].values  
        y_mortality = df['mother_alive_dead'].values
        
        # Create stratification variable combining outcomes
        strat_var = y_abort * 2 + (1 - y_mortality)  # 0: no_abort_alive, 1: no_abort_dead, 2: abort_alive, 3: abort_dead
        
        # Try group split by district first
        if 'district' in df.columns:
            districts = df['district'].values
            
            # Use GroupShuffleSplit
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            
            try:
                train_idx, val_idx = next(gss.split(df, strat_var, groups=districts))
            except ValueError:
                # Fall back to stratified split if group split fails
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                train_idx, val_idx = next(sss.split(df, strat_var))
        else:
            # Stratified split
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(sss.split(df, strat_var))
        
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        
        print(f"Train set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")
        
        # Check stratification
        print("\nTarget distribution in train set:")
        print(f"  Risk mean: {train_df['risk_to_woman_pct'].mean():.2f}%")
        print(f"  Abortion rate: {train_df['aborted'].mean():.3f}")
        print(f"  Mortality rate: {(1-train_df['mother_alive_dead']).mean():.4f}")
        
        print("\nTarget distribution in validation set:")
        print(f"  Risk mean: {val_df['risk_to_woman_pct'].mean():.2f}%")
        print(f"  Abortion rate: {val_df['aborted'].mean():.3f}")
        print(f"  Mortality rate: {(1-val_df['mother_alive_dead']).mean():.4f}")
        
        return train_df, val_df
    
    def prepare_features(self, df, fit_transformers=True):
        """Prepare final feature matrix"""
        df = df.copy()
        
        # Remove ID and target columns
        feature_cols = [col for col in df.columns if col not in 
                       ['patient_id', 'risk_to_woman_pct', 'aborted', 'mother_alive_dead']]
        
        X = df[feature_cols].copy()
        
        # Store feature names
        if fit_transformers:
            self.feature_names = X.columns.tolist()
        
        return X
    
    def preprocess_pipeline(self, filepath):
        """Complete preprocessing pipeline"""
        print("=== UTTARARISK-NEXT DATA PREPROCESSING PIPELINE ===")
        
        # Load data
        df = self.load_data(filepath)
        
        # Create clinical features
        print("\n1. Creating clinical feature buckets...")
        df = self.create_clinical_features(df)
        
        # Handle missing values
        print("2. Handling missing values with advanced imputation...")
        df = self.handle_missing_values(df)
        
        # Encode categorical variables
        print("3. Encoding categorical variables...")
        df = self.encode_categorical_variables(df)
        
        # Create train-validation split
        print("4. Creating stratified train-validation split...")
        train_df, val_df = self.create_train_val_split(df)
        
        # Prepare feature matrices
        print("5. Preparing final feature matrices...")
        X_train = self.prepare_features(train_df, fit_transformers=True)
        X_val = self.prepare_features(val_df, fit_transformers=False)
        
        # Extract targets
        y_train = {
            'risk': train_df['risk_to_woman_pct'].values,
            'abort': train_df['aborted'].values,
            'mortality': train_df['mother_alive_dead'].values
        }
        
        y_val = {
            'risk': val_df['risk_to_woman_pct'].values,
            'abort': val_df['aborted'].values,
            'mortality': val_df['mother_alive_dead'].values
        }
        
        print(f"\nFeature matrix shape: {X_train.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        
        # Save cleaned dataset
        cleaned_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        cleaned_df.to_csv('/home/sandbox/cleaned.csv', index=False)
        print("✓ Saved cleaned.csv")
        
        return X_train, X_val, y_train, y_val, train_df, val_df

# Run preprocessing
if __name__ == "__main__":
    preprocessor = UttarakhandDataPreprocessor()
    X_train, X_val, y_train, y_val, train_df, val_df = preprocessor.preprocess_pipeline('/home/sandbox/Data_Mother_1.csv')
    
    # Save preprocessed data
    np.savez('/home/sandbox/preprocessed_data.npz',
             X_train=X_train, X_val=X_val,
             y_train_risk=y_train['risk'], y_train_abort=y_train['abort'], y_train_mortality=y_train['mortality'],
             y_val_risk=y_val['risk'], y_val_abort=y_val['abort'], y_val_mortality=y_val['mortality'])
    
    print("\n✓ Preprocessing completed successfully!")
    print("✓ Saved preprocessed_data.npz")