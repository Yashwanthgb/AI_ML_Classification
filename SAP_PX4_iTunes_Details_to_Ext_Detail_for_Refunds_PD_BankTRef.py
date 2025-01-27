import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import json
import ast

class TransactionClassifier:
    def __init__(self, parquet_path):
        self.parquet_path = parquet_path
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.rule_name = 'SAP PX4 iTunes Details to Ext Detail for Refunds_PD_BankTRef'
        
    def load_and_filter_data(self):
        """Load data and filter for specific rule"""
        print("Loading data...")
        df = pd.read_parquet(self.parquet_path)
        
        # Extract match_rulename and filter
        df['match_rulename'] = df['match_res'].apply(lambda x: 
            ast.literal_eval(x)['match_rulename'] if isinstance(x, str) else x.get('match_rulename'))
        df = df[df['match_rulename'] == self.rule_name]
        
        # Extract match_status
        df['match_status'] = df['match_res'].apply(lambda x: 
            ast.literal_eval(x)['match_status'] if isinstance(x, str) else x.get('match_status'))
        
        print(f"Filtered data size: {len(df):,} records")
        return df
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        feature_columns = [
            'TransactionType',
            'TransactionRefNo',
            'CR_DR',
            'AuthCode',
            'AcquirerRefNumber'
        ]
        
        # Create feature DataFrame
        X = df[feature_columns].copy()
        y = (df['match_status'] == 'Matched').astype(int)
        
        print("\nFeature preparation:")
        print(f"Features shape: {X.shape}")
        print("Label distribution:")
        print(y.value_counts(normalize=True))
        
        # Handle missing values
        for col in X.columns:
            null_count = X[col].isnull().sum()
            if null_count > 0:
                print(f"Handling {null_count} null values in {col}")
                # Handle all columns as categorical since Amount is no longer a feature
                X[col] = X[col].fillna('Unknown')
        
        # Encode categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            print(f"Encoding {col}...")
            self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        return X, y
    
    def train_model(self, X, y):
        """Train the Random Forest model"""
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Training model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        print("\nModel Evaluation:")
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(importance)
        
        return {
            'model': self.model,
            'metrics': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': importance
        }
    
    def run_pipeline(self):
        """Run the complete classification pipeline"""
        print("Starting classification pipeline...")
        
        # Load and filter data
        df = self.load_and_filter_data()
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Train and evaluate model
        results = self.train_model(X, y)
        
        return results

def main():
    # Initialize and run pipeline
    parquet_path = "data_cache/batch_14238344.parquet"
    classifier = TransactionClassifier(parquet_path)
    results = classifier.run_pipeline()
    
    # Save results
    importance_df = results['feature_importance']
    importance_df.to_csv('feature_importance.csv', index=False)
    
    print("\nAnalysis complete! Results saved to feature_importance.csv")

if __name__ == "__main__":
    main()