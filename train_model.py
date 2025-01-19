import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from model_validation import ModelValidator
import talib
import optuna

class HybridIndexPredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_importance = None
        
    def load_and_prepare_data(self, csv_path):
        """
        Load data and perform feature engineering
        """
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate target
        df['hybrid_index'] = (df['eth_price'] / df['sp500_index']) * 1000
        
        # Feature Engineering
        df['eth_volatility'] = df['eth_price'].pct_change().rolling(window=7).std()
        df['sp500_volatility'] = df['sp500_index'].pct_change().rolling(window=7).std()
        df['price_ratio'] = df['eth_price'] / df['sp500_index']
        df['eth_ma7'] = df['eth_price'].rolling(window=7).mean()
        df['sp500_ma7'] = df['sp500_index'].rolling(window=7).mean()
        df['eth_momentum'] = df['eth_price'].pct_change(periods=7)
        df['sp500_momentum'] = df['sp500_index'].pct_change(periods=7)
        
        # Drop NaN values from feature engineering
        df = df.dropna()
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for modeling
        """
        feature_columns = [
            'eth_price', 'sp500_index', 'eth_volatility', 'sp500_volatility',
            'price_ratio', 'eth_ma7', 'sp500_ma7', 'eth_momentum', 'sp500_momentum'
        ]
        
        X = df[feature_columns]
        y = df['hybrid_index']
        
        return X, y
    
    def evaluate_models(self, X_train, X_test, y_train, y_test):
        """
        Evaluate XGBoost model with different configurations
        """
        models = {
            'xgboost_default': (xgb.XGBRegressor(random_state=self.random_state), {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'min_child_weight': [1, 3, 5]
            }),
            'xgboost_conservative': (xgb.XGBRegressor(random_state=self.random_state), {
                'n_estimators': [50, 100, 150],
                'max_depth': [2, 3, 4],
                'learning_rate': [0.005, 0.01, 0.05],
                'min_child_weight': [2, 4, 6],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            }),
            'xgboost_aggressive': (xgb.XGBRegressor(random_state=self.random_state), {
                'n_estimators': [300, 500, 700],
                'max_depth': [5, 7, 9],
                'learning_rate': [0.1, 0.2, 0.3],
                'min_child_weight': [1, 2, 3],
                'gamma': [0, 0.1, 0.2]
            })
        }
        
        best_score = float('-inf')
        best_model = None
        results = {}
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, (model, params) in models.items():
            print(f"\nTuning {name}...")
            grid_search = GridSearchCV(
                model, params, cv=tscv, scoring='r2',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            y_pred = grid_search.predict(X_test)
            score = r2_score(y_test, y_pred)
            results[name] = {
                'model': grid_search.best_estimator_,
                'params': grid_search.best_params_,
                'score': score,
                'predictions': y_pred
            }
            
            if score > best_score:
                best_score = score
                best_model = grid_search.best_estimator_
                self.best_model = best_model
        
        return results
    
    def analyze_feature_importance(self, X):
        """
        Analyze and plot feature importance
        """
        if hasattr(self.best_model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=importance)
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()
            
            self.feature_importance = importance
            return importance
    
    def plot_predictions(self, y_test, y_pred):
        """
        Plot actual vs predicted values
        """
        plt.figure(figsize=(12, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Hybrid Index')
        plt.ylabel('Predicted Hybrid Index')
        plt.title('Actual vs Predicted Hybrid Index')
        plt.tight_layout()
        plt.savefig('predictions.png')
        plt.close()
    
    def calculate_metrics(self, y_test, y_pred):
        """
        Calculate comprehensive model metrics
        """
        metrics = {
            'R² Score': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        return metrics
    
    def add_technical_indicators(self, df):
        """
        Ajoute des indicateurs techniques avancés en utilisant pandas
        """
        # Moyennes mobiles exponentielles
        df['eth_ema'] = df['eth_price'].ewm(span=14).mean()
        df['sp500_ema'] = df['sp500_index'].ewm(span=14).mean()
        
        # MACD simplifié
        exp1 = df['eth_price'].ewm(span=12).mean()
        exp2 = df['eth_price'].ewm(span=26).mean()
        df['eth_macd'] = exp1 - exp2
        df['eth_macd_signal'] = df['eth_macd'].ewm(span=9).mean()
        
        # RSI
        def calculate_rsi(series, periods=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df['eth_rsi'] = calculate_rsi(df['eth_price'])
        df['sp500_rsi'] = calculate_rsi(df['sp500_index'])
        
        # Bandes de Bollinger
        df['eth_sma'] = df['eth_price'].rolling(window=20).mean()
        df['eth_std'] = df['eth_price'].rolling(window=20).std()
        df['eth_bbands_upper'] = df['eth_sma'] + (df['eth_std'] * 2)
        df['eth_bbands_lower'] = df['eth_sma'] - (df['eth_std'] * 2)
        
        # Ratios et différences
        df['price_spread'] = df['eth_price'] - df['sp500_index']
        df['volatility_ratio'] = df['eth_volatility'] / df['sp500_volatility']
        
        return df
    
    def optimize_hyperparameters(self, X_train, y_train):
        """
        Optimisation bayésienne des hyperparamètres avec Optuna
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_loguniform('gamma', 1e-3, 1)
            }
            
            model = xgb.XGBRegressor(**params, random_state=self.random_state)
            
            # Validation croisée temporelle
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train = X_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_train = y_train.iloc[train_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                pred = model.predict(X_fold_val)
                score = r2_score(y_fold_val, pred)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
        return study.best_params

def main():
    # Configuration
    CSV_PATH = 'data.csv'
    MODEL_PATH = 'hybrid_model.pkl'
    RANDOM_STATE = 42
    
    try:
        # Initialize predictor
        predictor = HybridIndexPredictor(random_state=RANDOM_STATE)
        
        # Load and prepare data
        print("Loading and preparing data...")
        df = predictor.load_and_prepare_data(CSV_PATH)
        X, y = predictor.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        
        # Evaluate models
        print("Evaluating models...")
        results = predictor.evaluate_models(X_train, X_test, y_train, y_test)
        
        # Analyze best model
        best_model_name = max(results.items(), key=lambda x: x[1]['score'])[0]
        best_result = results[best_model_name]
        y_pred = best_result['predictions']
        
        # Calculate and display metrics
        metrics = predictor.calculate_metrics(y_test, y_pred)
        print("\nBest Model Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Analyze feature importance
        print("\nAnalyzing feature importance...")
        importance = predictor.analyze_feature_importance(X)
        print("\nTop 5 Most Important Features:")
        print(importance.head())
        
        # Plot predictions
        print("\nGenerating prediction plots...")
        predictor.plot_predictions(y_test, y_pred)
        
        # Save best model
        print(f"\nSaving best model ({best_model_name})...")
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(predictor.best_model, f)
        
        # Save model metadata
        metadata = {
            'model_type': best_model_name,
            'best_params': best_result['params'],
            'metrics': metrics,
            'feature_importance': importance.to_dict() if importance is not None else None,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 