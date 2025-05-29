"""
Advanced Machine Learning Models for 911 Emergency Calls Analysis

This module provides multiple ML models for different prediction tasks:
- Call volume forecasting
- Emergency type classification
- Response time prediction
- Geographic clustering
- Anomaly detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pickle
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from loguru import logger


class ModelTrainer:
    """Advanced machine learning model trainer"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the model trainer
        
        Args:
            config: Configuration dictionary for model parameters
        """
        self.config = config or self._default_config()
        self.models = {}
        self.results = {}
        self.best_models = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for model training"""
        return {
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 5,
            'scoring': 'accuracy',
            'n_jobs': -1,
            'optimize_hyperparameters': True,
            'save_models': True
        }
    
    def prepare_classification_data(self, df: pd.DataFrame, target_column: str = 'emergency_category') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for classification tasks
        
        Args:
            df: Processed DataFrame
            target_column: Target column for classification
            
        Returns:
            Tuple of (features, target, feature_names)
        """
        logger.info(f"Preparing classification data for target: {target_column}")
        
        # Select feature columns (exclude non-feature columns)
        exclude_columns = [
            'lat', 'lng', 'desc', 'title', 'timeStamp', 'addr', 'twp', 'zip',
            target_column, 'e'
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Handle missing values
        X = df[feature_columns].fillna(0)
        y = df[target_column]
        
        # Ensure all features are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        return X.values, y.values, feature_columns
    
    def prepare_regression_data(self, df: pd.DataFrame, target_column: str = 'hour') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for regression tasks
        
        Args:
            df: Processed DataFrame
            target_column: Target column for regression
            
        Returns:
            Tuple of (features, target, feature_names)
        """
        logger.info(f"Preparing regression data for target: {target_column}")
        
        # Create target variable (e.g., call volume per hour)
        if target_column == 'call_volume':
            # Group by hour and count calls
            hourly_calls = df.groupby(['year', 'month', 'day', 'hour']).size().reset_index(name='call_volume')
            
            # Create features from temporal data
            X = hourly_calls[['year', 'month', 'day', 'hour']].copy()
            
            # Add more features
            X['dayofweek'] = pd.to_datetime(hourly_calls[['year', 'month', 'day']]).dt.dayofweek
            X['is_weekend'] = X['dayofweek'].isin([5, 6]).astype(int)
            X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
            X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
            
            y = hourly_calls['call_volume']
            feature_columns = X.columns.tolist()
            
        else:
            # Use existing target column
            exclude_columns = [
                'lat', 'lng', 'desc', 'title', 'timeStamp', 'addr', 'twp', 'zip',
                target_column, 'e', 'emergency_category'
            ]
            
            feature_columns = [col for col in df.columns if col not in exclude_columns]
            X = df[feature_columns].fillna(0)
            y = df[target_column]
            
            # Ensure all features are numeric
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = pd.Categorical(X[col]).codes
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        return X.values, y.values, feature_columns
    
    def train_classification_models(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Train multiple classification models
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            
        Returns:
            Dictionary with model results
        """
        logger.info("Training classification models")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], stratify=y
        )
        
        # Define models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=self.config['random_state'], n_jobs=self.config['n_jobs']
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=self.config['random_state'], eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                random_state=self.config['random_state'], verbose=-1
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.config['random_state'], max_iter=1000
            )
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Evaluation metrics
                train_accuracy = accuracy_score(y_train, y_pred_train)
                test_accuracy = accuracy_score(y_test, y_pred_test)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=self.config['cv_folds'], 
                    scoring=self.config['scoring'], n_jobs=self.config['n_jobs']
                )
                
                results[name] = {
                    'model': model,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'classification_report': classification_report(y_test, y_pred_test),
                    'feature_importance': self._get_feature_importance(model, feature_names)
                }
                
                logger.info(f"{name} - Test Accuracy: {test_accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
        self.best_models['classification'] = {
            'name': best_model_name,
            'model': results[best_model_name]['model'],
            'performance': results[best_model_name]
        }
        
        logger.info(f"Best classification model: {best_model_name}")
        return results
    
    def train_regression_models(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Train multiple regression models
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            
        Returns:
            Dictionary with model results
        """
        logger.info("Training regression models")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state']
        )
        
        # Define models
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, random_state=self.config['random_state'], n_jobs=self.config['n_jobs']
            ),
            'xgboost': xgb.XGBRegressor(
                random_state=self.config['random_state']
            ),
            'lightgbm': lgb.LGBMRegressor(
                random_state=self.config['random_state'], verbose=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                random_state=self.config['random_state']
            ),
            'ridge': Ridge(alpha=1.0)
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Evaluation metrics
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=self.config['cv_folds'], 
                    scoring='neg_mean_absolute_error', n_jobs=self.config['n_jobs']
                )
                
                results[name] = {
                    'model': model,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'cv_mae': -cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'feature_importance': self._get_feature_importance(model, feature_names)
                }
                
                logger.info(f"{name} - Test MAE: {test_mae:.4f}, Test R²: {test_r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue
        
        # Find best model (lowest test MAE)
        best_model_name = min(results.keys(), key=lambda x: results[x]['test_mae'])
        self.best_models['regression'] = {
            'name': best_model_name,
            'model': results[best_model_name]['model'],
            'performance': results[best_model_name]
        }
        
        logger.info(f"Best regression model: {best_model_name}")
        return results
    
    def train_time_series_models(self, df: pd.DataFrame) -> Dict:
        """
        Train time series forecasting models
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            Dictionary with time series model results
        """
        logger.info("Training time series models")
        
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available, skipping time series models")
            return {}
        
        # Prepare time series data
        ts_data = df.groupby(df['timeStamp'].dt.date).size().reset_index()
        ts_data.columns = ['ds', 'y']
        ts_data['ds'] = pd.to_datetime(ts_data['ds'])
        
        # Split data (80-20 split)
        split_point = int(len(ts_data) * 0.8)
        train_data = ts_data[:split_point]
        test_data = ts_data[split_point:]
        
        results = {}
        
        try:
            # Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            model.fit(train_data)
            
            # Make predictions
            future = model.make_future_dataframe(periods=len(test_data))
            forecast = model.predict(future)
            
            # Evaluate
            train_forecast = forecast[:split_point]
            test_forecast = forecast[split_point:]
            
            train_mae = mean_absolute_error(train_data['y'], train_forecast['yhat'])
            test_mae = mean_absolute_error(test_data['y'], test_forecast['yhat'])
            
            results['prophet'] = {
                'model': model,
                'forecast': forecast,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'mape': np.mean(np.abs((test_data['y'] - test_forecast['yhat']) / test_data['y'])) * 100
            }
            
            self.best_models['time_series'] = {
                'name': 'prophet',
                'model': model,
                'performance': results['prophet']
            }
            
            logger.info(f"Prophet - Test MAE: {test_mae:.4f}, MAPE: {results['prophet']['mape']:.2f}%")
            
        except Exception as e:
            logger.error(f"Error training Prophet model: {str(e)}")
        
        return results
    
    def train_clustering_models(self, X: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Train clustering models for pattern discovery
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary with clustering results
        """
        logger.info("Training clustering models")
        
        # Standardize features for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {}
        
        # K-Means clustering
        try:
            # Find optimal number of clusters using elbow method
            inertias = []
            k_range = range(2, 11)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=self.config['random_state'])
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
            
            # Use k=5 as default (or find elbow point)
            optimal_k = 5
            kmeans = KMeans(n_clusters=optimal_k, random_state=self.config['random_state'])
            clusters = kmeans.fit_predict(X_scaled)
            
            results['kmeans'] = {
                'model': kmeans,
                'clusters': clusters,
                'n_clusters': optimal_k,
                'inertias': inertias,
                'silhouette_score': self._calculate_silhouette_score(X_scaled, clusters)
            }
            
            logger.info(f"K-Means clustering completed with {optimal_k} clusters")
            
        except Exception as e:
            logger.error(f"Error in K-Means clustering: {str(e)}")
        
        # DBSCAN clustering
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(X_scaled)
            
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)
            
            results['dbscan'] = {
                'model': dbscan,
                'clusters': clusters,
                'n_clusters': n_clusters,
                'n_noise': n_noise
            }
            
            logger.info(f"DBSCAN clustering: {n_clusters} clusters, {n_noise} noise points")
            
        except Exception as e:
            logger.error(f"Error in DBSCAN clustering: {str(e)}")
        
        return results
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict:
        """Get feature importance from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
            else:
                return {}
            
            feature_importance = dict(zip(feature_names, importance))
            # Sort by importance
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
        except Exception:
            return {}
    
    def _calculate_silhouette_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering"""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(X, labels)
        except Exception:
            return 0.0
    
    def hyperparameter_optimization(self, model_type: str, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Perform hyperparameter optimization
        
        Args:
            model_type: Type of model ('classification' or 'regression')
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with optimized parameters
        """
        logger.info(f"Performing hyperparameter optimization for {model_type}")
        
        if not self.config['optimize_hyperparameters']:
            return {}
        
        # Define parameter grids
        if model_type == 'classification':
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                },
                'xgboost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            }
            scoring = 'accuracy'
            
        else:  # regression
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                },
                'xgboost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            }
            scoring = 'neg_mean_absolute_error'
        
        optimized_results = {}
        
        for model_name, param_grid in param_grids.items():
            try:
                if model_name == 'random_forest':
                    if model_type == 'classification':
                        base_model = RandomForestClassifier(random_state=self.config['random_state'])
                    else:
                        base_model = RandomForestRegressor(random_state=self.config['random_state'])
                elif model_name == 'xgboost':
                    if model_type == 'classification':
                        base_model = xgb.XGBClassifier(random_state=self.config['random_state'])
                    else:
                        base_model = xgb.XGBRegressor(random_state=self.config['random_state'])
                
                grid_search = GridSearchCV(
                    base_model, param_grid, cv=3, scoring=scoring,
                    n_jobs=self.config['n_jobs'], verbose=0
                )
                
                grid_search.fit(X, y)
                
                optimized_results[model_name] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'best_model': grid_search.best_estimator_
                }
                
                logger.info(f"{model_name} optimization completed. Best score: {grid_search.best_score_:.4f}")
                
            except Exception as e:
                logger.error(f"Error optimizing {model_name}: {str(e)}")
                continue
        
        return optimized_results
    
    def save_models(self, models_dict: Dict, output_dir: str) -> None:
        """
        Save trained models to disk
        
        Args:
            models_dict: Dictionary containing trained models
            output_dir: Directory to save models
        """
        if not self.config['save_models']:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for model_type, models in models_dict.items():
            for model_name, model_info in models.items():
                if 'model' in model_info:
                    model_path = output_path / f"{model_type}_{model_name}.pkl"
                    
                    try:
                        joblib.dump(model_info['model'], model_path)
                        logger.info(f"Saved {model_type} {model_name} to {model_path}")
                    except Exception as e:
                        logger.error(f"Error saving {model_type} {model_name}: {str(e)}")
    
    def generate_model_report(self, results: Dict) -> str:
        """
        Generate comprehensive model performance report
        
        Args:
            results: Dictionary with model results
            
        Returns:
            Formatted report string
        """
        report = "="*50 + "\n"
        report += "MODEL PERFORMANCE REPORT\n"
        report += "="*50 + "\n\n"
        
        for model_type, models in results.items():
            report += f"{model_type.upper()} MODELS\n"
            report += "-"*30 + "\n"
            
            for model_name, metrics in models.items():
                report += f"\n{model_name.title()}:\n"
                
                if model_type == 'classification':
                    report += f"  Test Accuracy: {metrics.get('test_accuracy', 0):.4f}\n"
                    report += f"  CV Score: {metrics.get('cv_mean', 0):.4f} (+/- {metrics.get('cv_std', 0):.4f})\n"
                
                elif model_type == 'regression':
                    report += f"  Test MAE: {metrics.get('test_mae', 0):.4f}\n"
                    report += f"  Test R²: {metrics.get('test_r2', 0):.4f}\n"
                    report += f"  CV MAE: {metrics.get('cv_mae', 0):.4f}\n"
                
                elif model_type == 'time_series':
                    report += f"  Test MAE: {metrics.get('test_mae', 0):.4f}\n"
                    report += f"  MAPE: {metrics.get('mape', 0):.2f}%\n"
            
            report += "\n"
        
        return report


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train machine learning models for 911 calls analysis')
    parser.add_argument('--data', required=True, help='Processed data file path')
    parser.add_argument('--output', required=True, help='Output directory for models')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--models', nargs='+', default=['classification', 'regression', 'clustering'],
                       choices=['classification', 'regression', 'time_series', 'clustering'],
                       help='Models to train')
    
    args = parser.parse_args()
    
    # Load data
    if args.data.endswith('.parquet'):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)
    
    # Initialize trainer
    trainer = ModelTrainer()
    all_results = {}
    
    # Train requested models
    if 'classification' in args.models:
        X, y, feature_names = trainer.prepare_classification_data(df)
        classification_results = trainer.train_classification_models(X, y, feature_names)
        all_results['classification'] = classification_results
    
    if 'regression' in args.models:
        X, y, feature_names = trainer.prepare_regression_data(df, target_column='call_volume')
        regression_results = trainer.train_regression_models(X, y, feature_names)
        all_results['regression'] = regression_results
    
    if 'time_series' in args.models and PROPHET_AVAILABLE:
        ts_results = trainer.train_time_series_models(df)
        all_results['time_series'] = ts_results
    
    if 'clustering' in args.models:
        X, y, feature_names = trainer.prepare_classification_data(df)
        clustering_results = trainer.train_clustering_models(X, feature_names)
        all_results['clustering'] = clustering_results
    
    # Save models
    trainer.save_models(all_results, args.output)
    
    # Generate and save report
    report = trainer.generate_model_report(all_results)
    report_path = Path(args.output) / 'model_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    logger.info(f"Training completed! Models saved to {args.output}")


if __name__ == "__main__":
    main() 