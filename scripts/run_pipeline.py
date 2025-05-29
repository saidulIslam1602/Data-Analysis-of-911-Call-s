#!/usr/bin/env python3
"""
911 Emergency Calls Analytics Pipeline Orchestrator

This script provides a command-line interface to run the complete data science
pipeline including data processing, feature engineering, model training,
evaluation, and dashboard deployment.
"""

import sys
import os
from pathlib import Path
import argparse
import subprocess
import time
from datetime import datetime
from typing import List, Dict, Optional
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from loguru import logger
from utils.config import get_config, ConfigManager
from data.processor import EmergencyCallsProcessor
from models.train import ModelTrainer
from visualization.plots import create_publication_plots


class PipelineOrchestrator:
    """Orchestrates the complete data science pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline orchestrator
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.start_time = datetime.now()
        
        # Setup logging
        self._setup_logging()
        
        logger.info("🚨 911 Emergency Calls Analytics Pipeline Started")
        logger.info(f"Configuration loaded from: {config_path or 'default'}")
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.logging
        
        # Create logs directory
        log_dir = Path(log_config.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure loguru
        logger.remove()  # Remove default handler
        logger.add(
            sys.stdout,
            level=log_config.level,
            format=log_config.format
        )
        logger.add(
            log_config.log_file,
            level=log_config.level,
            format=log_config.format,
            rotation=log_config.rotation,
            retention=log_config.retention
        )
    
    def check_requirements(self) -> bool:
        """
        Check if all requirements are met
        
        Returns:
            True if all requirements are met
        """
        logger.info("🔍 Checking requirements...")
        
        # Check if data file exists
        data_path = Path(self.config.data.raw_data_path)
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return False
        
        logger.info(f"✅ Data file found: {data_path}")
        
        # Check required directories
        required_dirs = [
            "data/processed",
            "models",
            "plots",
            "reports",
            "logs"
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Directory ready: {dir_path}")
        
        # Check Python packages
        required_packages = [
            "pandas", "numpy", "scikit-learn", "xgboost", "lightgbm",
            "plotly", "streamlit", "folium", "loguru"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
            logger.info("Please install missing packages using: pip install -r requirements.txt")
            return False
        
        logger.info("✅ All Python packages available")
        return True
    
    def run_data_processing(self) -> str:
        """
        Run data processing pipeline
        
        Returns:
            Path to processed data file
        """
        logger.info("📊 Starting data processing...")
        
        try:
            # Initialize processor
            processor = EmergencyCallsProcessor(config=self.config.data.__dict__)
            
            # Load and validate data
            logger.info("Loading raw data...")
            df_raw = processor.load_and_validate_data(self.config.data.raw_data_path)
            logger.info(f"Loaded {len(df_raw):,} raw records")
            
            # Clean data
            logger.info("Cleaning data...")
            df_clean = processor.clean_data(df_raw)
            logger.info(f"After cleaning: {len(df_clean):,} records")
            
            # Engineer features
            logger.info("Engineering features...")
            df_features = processor.engineer_features(df_clean)
            logger.info(f"Created {len(processor.feature_columns)} new features")
            
            # Prepare for modeling
            logger.info("Preparing data for modeling...")
            df_model, preprocessing_info = processor.prepare_for_modeling(df_features)
            
            # Save processed data
            processed_dir = Path(self.config.data.processed_data_path)
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = processed_dir / "processed_data.parquet"
            processor.save_processed_data(df_model, str(output_file))
            
            # Save preprocessing info
            import pickle
            with open(processed_dir / "preprocessing_info.pkl", "wb") as f:
                pickle.dump(preprocessing_info, f)
            
            logger.info(f"✅ Data processing completed: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"❌ Data processing failed: {str(e)}")
            raise
    
    def run_model_training(self, data_path: str) -> Dict[str, str]:
        """
        Run model training pipeline
        
        Args:
            data_path: Path to processed data
            
        Returns:
            Dictionary with paths to trained models
        """
        logger.info("🤖 Starting model training...")
        
        try:
            # Load processed data
            import pandas as pd
            df = pd.read_parquet(data_path)
            logger.info(f"Loaded {len(df):,} processed records for training")
            
            # Initialize trainer
            trainer = ModelTrainer(config=self.config.model.__dict__)
            
            all_results = {}
            model_paths = {}
            
            # Train classification models
            try:
                enable_ml_models = getattr(self.config, 'feature_flags', {}).get('enable_ml_models', True)
            except AttributeError:
                enable_ml_models = True
                
            if enable_ml_models:
                logger.info("Training classification models...")
                X, y, feature_names = trainer.prepare_classification_data(df)
                classification_results = trainer.train_classification_models(X, y, feature_names)
                all_results['classification'] = classification_results
                logger.info("✅ Classification models trained")
            
            # Train regression models
            try:
                enable_ml_models = getattr(self.config, 'feature_flags', {}).get('enable_ml_models', True)
            except AttributeError:
                enable_ml_models = True
                
            if enable_ml_models:
                logger.info("Training regression models...")
                # Create a simple target for regression (call count per hour)
                df_regression = df.copy()
                df_regression['call_volume'] = 1  # Each row represents one call
                X, y, feature_names = trainer.prepare_regression_data(df_regression, target_column='call_volume')
                regression_results = trainer.train_regression_models(X, y, feature_names)
                all_results['regression'] = regression_results
                logger.info("✅ Regression models trained")
            
            # Train time series models
            try:
                enable_ts = getattr(self.config, 'feature_flags', {}).get('enable_time_series_forecasting', True)
            except AttributeError:
                enable_ts = True
                
            if enable_ts:
                logger.info("Training time series models...")
                ts_results = trainer.train_time_series_models(df)
                all_results['time_series'] = ts_results
                logger.info("✅ Time series models trained")
            
            # Train clustering models
            try:
                enable_clustering = getattr(self.config, 'feature_flags', {}).get('enable_clustering', True)
            except AttributeError:
                enable_clustering = True
                
            if enable_clustering:
                logger.info("Training clustering models...")
                X, y, feature_names = trainer.prepare_classification_data(df)
                clustering_results = trainer.train_clustering_models(X, feature_names)
                all_results['clustering'] = clustering_results
                logger.info("✅ Clustering models trained")
            
            # Save models
            models_dir = Path(self.config.model.model_output_path)
            models_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_models(all_results, str(models_dir))
            
            # Generate model report
            report = trainer.generate_model_report(all_results)
            report_path = models_dir / "model_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            
            logger.info(f"✅ Model training completed. Report saved: {report_path}")
            
            model_paths = {
                'models_directory': str(models_dir),
                'report_path': str(report_path)
            }
            
            return model_paths
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {str(e)}")
            raise
    
    def run_visualization(self, data_path: str) -> str:
        """
        Generate visualizations and plots
        
        Args:
            data_path: Path to processed data
            
        Returns:
            Path to plots directory
        """
        logger.info("📈 Generating visualizations...")
        
        try:
            import pandas as pd
            df = pd.read_parquet(data_path)
            
            plots_dir = Path("plots")
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            create_publication_plots(df, str(plots_dir))
            
            logger.info(f"✅ Visualizations generated: {plots_dir}")
            return str(plots_dir)
            
        except Exception as e:
            logger.error(f"❌ Visualization generation failed: {str(e)}")
            raise
    
    def deploy_dashboard(self, background: bool = True) -> Dict[str, str]:
        """
        Deploy the Streamlit dashboard
        
        Args:
            background: Whether to run dashboard in background
            
        Returns:
            Dashboard deployment information
        """
        logger.info("🚀 Deploying dashboard...")
        
        try:
            dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"
            
            if not dashboard_path.exists():
                raise FileNotFoundError(f"Dashboard file not found: {dashboard_path}")
            
            # Streamlit command
            cmd = [
                "streamlit", "run", str(dashboard_path),
                "--server.port", str(self.config.dashboard.port),
                "--server.address", self.config.dashboard.host
            ]
            
            if background:
                # Run in background
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Give it a moment to start
                time.sleep(3)
                
                if process.poll() is None:
                    dashboard_url = f"http://{self.config.dashboard.host}:{self.config.dashboard.port}"
                    logger.info(f"✅ Dashboard deployed: {dashboard_url}")
                    logger.info(f"Dashboard process PID: {process.pid}")
                    
                    return {
                        'url': dashboard_url,
                        'pid': str(process.pid),
                        'status': 'running'
                    }
                else:
                    stdout, stderr = process.communicate()
                    logger.error(f"Dashboard failed to start: {stderr}")
                    raise RuntimeError(f"Dashboard deployment failed: {stderr}")
            else:
                # Run in foreground
                logger.info("Starting dashboard in foreground mode...")
                result = subprocess.run(cmd)
                return {'status': 'completed', 'return_code': result.returncode}
                
        except Exception as e:
            logger.error(f"❌ Dashboard deployment failed: {str(e)}")
            raise
    
    def run_full_pipeline(self, deploy_dashboard: bool = True) -> Dict[str, str]:
        """
        Run the complete pipeline
        
        Args:
            deploy_dashboard: Whether to deploy dashboard after training
            
        Returns:
            Pipeline execution results
        """
        logger.info("🔄 Running full pipeline...")
        
        results = {}
        
        try:
            # Check requirements
            if not self.check_requirements():
                raise RuntimeError("Requirements check failed")
            
            # Data processing
            data_path = self.run_data_processing()
            results['data_path'] = data_path
            
            # Model training
            model_info = self.run_model_training(data_path)
            results.update(model_info)
            
            # Visualization
            plots_path = self.run_visualization(data_path)
            results['plots_path'] = plots_path
            
            # Deploy dashboard
            if deploy_dashboard:
                dashboard_info = self.deploy_dashboard(background=True)
                results.update(dashboard_info)
            
            # Calculate execution time
            execution_time = datetime.now() - self.start_time
            results['execution_time'] = str(execution_time)
            
            logger.info(f"✅ Full pipeline completed in {execution_time}")
            logger.info("🎉 911 Emergency Calls Analytics Platform is ready!")
            
            if 'url' in results:
                logger.info(f"🌐 Access the dashboard at: {results['url']}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {str(e)}")
            raise
    
    def cleanup(self):
        """Cleanup resources and temporary files"""
        logger.info("🧹 Cleaning up...")
        
        # Add cleanup logic here if needed
        logger.info("✅ Cleanup completed")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="911 Emergency Calls Analytics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --full                    # Run complete pipeline
  python run_pipeline.py --data-only               # Process data only
  python run_pipeline.py --train-only              # Train models only
  python run_pipeline.py --dashboard-only          # Deploy dashboard only
  python run_pipeline.py --config config/prod.yaml # Use custom config
        """
    )
    
    # Pipeline options
    parser.add_argument('--full', action='store_true', help='Run complete pipeline')
    parser.add_argument('--data-only', action='store_true', help='Run data processing only')
    parser.add_argument('--train-only', action='store_true', help='Run model training only')
    parser.add_argument('--viz-only', action='store_true', help='Generate visualizations only')
    parser.add_argument('--dashboard-only', action='store_true', help='Deploy dashboard only')
    
    # Configuration
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--data-path', help='Path to processed data (for train-only mode)')
    
    # Dashboard options
    parser.add_argument('--no-dashboard', action='store_true', help='Skip dashboard deployment')
    parser.add_argument('--foreground', action='store_true', help='Run dashboard in foreground')
    
    # Utility options
    parser.add_argument('--check-only', action='store_true', help='Check requirements only')
    parser.add_argument('--version', action='version', version='1.0.0')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(config_path=args.config)
    
    try:
        if args.check_only:
            success = orchestrator.check_requirements()
            sys.exit(0 if success else 1)
        
        elif args.data_only:
            data_path = orchestrator.run_data_processing()
            print(f"Processed data saved to: {data_path}")
        
        elif args.train_only:
            if not args.data_path:
                logger.error("--data-path required for train-only mode")
                sys.exit(1)
            model_info = orchestrator.run_model_training(args.data_path)
            print(f"Models saved to: {model_info['models_directory']}")
        
        elif args.viz_only:
            if not args.data_path:
                logger.error("--data-path required for viz-only mode")
                sys.exit(1)
            plots_path = orchestrator.run_visualization(args.data_path)
            print(f"Plots saved to: {plots_path}")
        
        elif args.dashboard_only:
            dashboard_info = orchestrator.deploy_dashboard(background=not args.foreground)
            if 'url' in dashboard_info:
                print(f"Dashboard available at: {dashboard_info['url']}")
        
        elif args.full:
            results = orchestrator.run_full_pipeline(deploy_dashboard=not args.no_dashboard)
            
            print("\n" + "="*50)
            print("🎉 PIPELINE EXECUTION COMPLETED")
            print("="*50)
            print(f"Data processed: {results.get('data_path', 'N/A')}")
            print(f"Models trained: {results.get('models_directory', 'N/A')}")
            print(f"Visualizations: {results.get('plots_path', 'N/A')}")
            print(f"Execution time: {results.get('execution_time', 'N/A')}")
            
            if 'url' in results:
                print(f"Dashboard URL: {results['url']}")
            
            print("="*50)
        
        else:
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        orchestrator.cleanup()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        orchestrator.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main() 