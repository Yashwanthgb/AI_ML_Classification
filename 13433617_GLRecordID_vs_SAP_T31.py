import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import json
import ast
from datetime import datetime
import os
import joblib

class TransactionClassifier:
    def __init__(self, parquet_path):
        self.parquet_path = parquet_path
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.rule_name = 'GLRecordID vs SAP T31'
        
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
            'GLRecordID',
            'GLStatus',
            'TransactionType',
            'DocType'
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
                X[col] = X[col].fillna('Unknown')
        
        # Encode categorical variables
        for col in X.columns:
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
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'model': self.model,
            'metrics': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': importance,
            'confusion_matrix': conf_matrix,
            'test_data': (X_test, y_test)
        }
    
    def save_model(self, results, output_dir='model_outputs/gl_record_classifier'):
        """Save model artifacts and results"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model artifacts
        artifact_paths = {}
        
        # Save model
        model_path = os.path.join(output_dir, f'model_{timestamp}.joblib')
        joblib.dump(self.model, model_path)
        artifact_paths['model'] = model_path
        
        # Save label encoders
        encoders_path = os.path.join(output_dir, f'label_encoders_{timestamp}.joblib')
        joblib.dump(self.label_encoders, encoders_path)
        artifact_paths['encoders'] = encoders_path
        
        # Save metrics
        metrics_path = os.path.join(output_dir, f'metrics_{timestamp}.json')
        with open(metrics_path, 'w') as f:
            json.dump(results['metrics'], f, indent=4)
        artifact_paths['metrics'] = metrics_path
        
        # Save feature importance
        importance_path = os.path.join(output_dir, f'feature_importance_{timestamp}.csv')
        results['feature_importance'].to_csv(importance_path, index=False)
        artifact_paths['feature_importance'] = importance_path
        
        # Save confusion matrix
        conf_matrix_path = os.path.join(output_dir, f'confusion_matrix_{timestamp}.npy')
        np.save(conf_matrix_path, results['confusion_matrix'])
        artifact_paths['confusion_matrix'] = conf_matrix_path
        
        # Save model summary
        summary_path = os.path.join(output_dir, f'model_summary_{timestamp}.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Model Summary - {self.rule_name}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Features used: {list(results['feature_importance']['feature'])}\n\n")
            f.write("Model Performance:\n")
            f.write("-"*20 + "\n")
            f.write(f"Accuracy: {results['metrics']['accuracy']:.4f}\n")
            f.write(f"Weighted F1-score: {results['metrics']['weighted avg']['f1-score']:.4f}\n")
        
        print(f"\nModel artifacts saved in {output_dir}")
        return artifact_paths
    
    def run_pipeline(self):
        """Run the complete classification pipeline"""
        print("Starting classification pipeline...")
        print(f"Rule name: {self.rule_name}")
        
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
    parquet_path = "data_cache/batch_13433617.parquet"  # Updated batch ID
    classifier = TransactionClassifier(parquet_path)
    results, artifact_paths = classifier.run_pipeline()
    
    # Print summary
    print("\nPipeline Summary:")
    print("="*50)
    print(f"Model saved to: {artifact_paths['model']}")
    print(f"Feature importance saved to: {artifact_paths['feature_importance']}")
    print(f"Model metrics saved to: {artifact_paths['metrics']}")
    
    print("\nTop Features by Importance:")
    print(results['feature_importance'].head())
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    print("\nModel Performance Summary:")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Weighted F1-score: {results['metrics']['weighted avg']['f1-score']:.4f}")

if __name__ == "__main__":
    main()