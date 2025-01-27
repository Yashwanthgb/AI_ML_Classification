import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import json
import ast
from datetime import datetime

class TransactionClassifier:
    def __init__(self, parquet_path):
        self.parquet_path = parquet_path
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.rule_name = 'Ext Settlement Summary_Add 1 day vs Payment Detail_Refund'
        
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
            'BankTrfRef',
            'CR_DR',
            'PostDate'
        ]
        
        # Create feature DataFrame
        X = df[feature_columns].copy()
        y = (df['match_status'] == 'Matched').astype(int)
        
        print("\nFeature preparation:")
        print(f"Features shape: {X.shape}")
        print("Label distribution:")
        print(y.value_counts(normalize=True))
        
        # Convert PostDate to datetime features
        X['PostDate'] = pd.to_datetime(X['PostDate'])
        X['post_day_of_week'] = X['PostDate'].dt.dayofweek
        X['post_day_of_month'] = X['PostDate'].dt.day
        X['post_month'] = X['PostDate'].dt.month
        X['post_year'] = X['PostDate'].dt.year
        
        # Drop original PostDate column
        X = X.drop('PostDate', axis=1)
        
        # Handle missing values
        for col in X.columns:
            null_count = X[col].isnull().sum()
            if null_count > 0:
                print(f"Handling {null_count} null values in {col}")
                if col in ['post_day_of_week', 'post_day_of_month', 'post_month', 'post_year']:
                    X[col] = X[col].fillna(-1)
                else:
                    X[col] = X[col].fillna('Unknown')
        
        # Encode categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            print(f"Encoding {col}...")
            self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        # Scale numerical date features
        date_cols = ['post_day_of_week', 'post_day_of_month', 'post_month', 'post_year']
        X[date_cols] = self.scaler.fit_transform(X[date_cols])
        
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
            'feature_importance': importance,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def save_model(self, results, output_dir='model_outputs'):
        """Save model artifacts and results"""
        import os
        import joblib
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = os.path.join(output_dir, f'model_{timestamp}.joblib')
        joblib.dump(self.model, model_path)
        
        # Save label encoders
        encoders_path = os.path.join(output_dir, f'label_encoders_{timestamp}.joblib')
        joblib.dump(self.label_encoders, encoders_path)
        
        # Save scaler
        scaler_path = os.path.join(output_dir, f'scaler_{timestamp}.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature importance
        importance_path = os.path.join(output_dir, f'feature_importance_{timestamp}.csv')
        results['feature_importance'].to_csv(importance_path, index=False)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, f'metrics_{timestamp}.json')
        with open(metrics_path, 'w') as f:
            json.dump(results['metrics'], f, indent=4)
        
        print(f"\nModel artifacts saved in {output_dir}")
        return {
            'model_path': model_path,
            'encoders_path': encoders_path,
            'scaler_path': scaler_path,
            'importance_path': importance_path,
            'metrics_path': metrics_path
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
        
        # Save model artifacts
        artifact_paths = self.save_model(results)
        
        return results, artifact_paths

def main():
    # Initialize and run pipeline
    parquet_path = "data_cache/batch_14238344.parquet"
    classifier = TransactionClassifier(parquet_path)
    results, artifact_paths = classifier.run_pipeline()
    
    # Print summary
    print("\nPipeline Summary:")
    print("----------------")
    print(f"Model saved to: {artifact_paths['model_path']}")
    print(f"Feature importance saved to: {artifact_paths['importance_path']}")
    print(f"Model metrics saved to: {artifact_paths['metrics_path']}")
    print("\nTop 5 most important features:")
    print(results['feature_importance'].head())
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])

if __name__ == "__main__":
    main()