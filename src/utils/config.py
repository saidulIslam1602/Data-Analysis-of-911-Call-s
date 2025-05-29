"""
Configuration Management Module

Handles all configuration settings for the 911 Emergency Calls Analytics Platform.
Supports environment variables, YAML files, and command-line arguments.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging


@dataclass
class DataConfig:
    """Data processing configuration"""
    raw_data_path: str = "data/raw/911.csv"
    processed_data_path: str = "data/processed/"
    chunk_size: int = 10000
    coordinate_bounds: Dict[str, float] = field(default_factory=lambda: {
        'lat_min': 39.0, 'lat_max': 42.0,
        'lng_min': -76.0, 'lng_max': -74.0
    })
    outlier_threshold: float = 3.0
    min_call_frequency: int = 5


@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_jobs: int = -1
    optimize_hyperparameters: bool = True
    save_models: bool = True
    model_output_path: str = "models/"
    
    # XGBoost parameters
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    })
    
    # Random Forest parameters
    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    })


@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    host: str = "localhost"
    port: int = 8501
    debug: bool = False
    cache_ttl: int = 3600  # Cache time-to-live in seconds
    max_map_points: int = 5000
    default_date_range_days: int = 365


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
    log_file: str = "logs/app.log"
    rotation: str = "1 week"
    retention: str = "1 month"


@dataclass
class DatabaseConfig:
    """Database configuration (for future use)"""
    host: str = "localhost"
    port: int = 5432
    database: str = "emergency_calls"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 5


@dataclass
class APIConfig:
    """API configuration (for future use)"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class AppConfig:
    """Main application configuration"""
    app_name: str = "911 Emergency Calls Analytics Platform"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)


class ConfigManager:
    """Configuration manager with support for YAML files and environment variables"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self._config = None
        
    def load_config(self) -> AppConfig:
        """
        Load configuration from file and environment variables
        
        Returns:
            AppConfig object with all settings
        """
        if self._config is not None:
            return self._config
        
        # Start with default configuration
        config_dict = {}
        
        # Load from YAML file if provided
        if self.config_path and Path(self.config_path).exists():
            config_dict = self._load_yaml_config(self.config_path)
        
        # Override with environment variables
        config_dict = self._load_env_config(config_dict)
        
        # Create AppConfig object
        self._config = self._dict_to_config(config_dict)
        
        return self._config
    
    def _load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config or {}
        except Exception as e:
            logging.warning(f"Could not load config file {config_path}: {e}")
            return {}
    
    def _load_env_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        # Environment variable mappings
        env_mappings = {
            'APP_NAME': ('app_name',),
            'APP_VERSION': ('version',),
            'APP_ENVIRONMENT': ('environment',),
            'APP_DEBUG': ('debug',),
            
            # Data configuration
            'DATA_RAW_PATH': ('data', 'raw_data_path'),
            'DATA_PROCESSED_PATH': ('data', 'processed_data_path'),
            'DATA_CHUNK_SIZE': ('data', 'chunk_size'),
            
            # Model configuration
            'MODEL_TEST_SIZE': ('model', 'test_size'),
            'MODEL_RANDOM_STATE': ('model', 'random_state'),
            'MODEL_CV_FOLDS': ('model', 'cv_folds'),
            'MODEL_N_JOBS': ('model', 'n_jobs'),
            
            # Dashboard configuration
            'DASHBOARD_HOST': ('dashboard', 'host'),
            'DASHBOARD_PORT': ('dashboard', 'port'),
            'DASHBOARD_DEBUG': ('dashboard', 'debug'),
            
            # Database configuration
            'DB_HOST': ('database', 'host'),
            'DB_PORT': ('database', 'port'),
            'DB_NAME': ('database', 'database'),
            'DB_USER': ('database', 'username'),
            'DB_PASSWORD': ('database', 'password'),
            
            # API configuration
            'API_HOST': ('api', 'host'),
            'API_PORT': ('api', 'port'),
            'API_WORKERS': ('api', 'workers'),
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                env_value = self._convert_env_value(env_value)
                
                # Set nested configuration value
                current = config_dict
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[config_path[-1]] = env_value
        
        return config_dict
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig object"""
        try:
            # Create sub-configurations
            data_config = DataConfig(**config_dict.get('data', {}))
            model_config = ModelConfig(**config_dict.get('model', {}))
            dashboard_config = DashboardConfig(**config_dict.get('dashboard', {}))
            logging_config = LoggingConfig(**config_dict.get('logging', {}))
            database_config = DatabaseConfig(**config_dict.get('database', {}))
            api_config = APIConfig(**config_dict.get('api', {}))
            
            # Remove sub-config dicts from main config
            main_config = {k: v for k, v in config_dict.items() 
                          if k not in ['data', 'model', 'dashboard', 'logging', 'database', 'api']}
            
            # Create main configuration
            app_config = AppConfig(
                **main_config,
                data=data_config,
                model=model_config,
                dashboard=dashboard_config,
                logging=logging_config,
                database=database_config,
                api=api_config
            )
            
            return app_config
            
        except Exception as e:
            logging.warning(f"Error creating config object: {e}")
            return AppConfig()
    
    def save_config(self, config: AppConfig, output_path: str) -> None:
        """
        Save configuration to YAML file
        
        Args:
            config: AppConfig object to save
            output_path: Path to save the configuration file
        """
        try:
            # Convert config object to dictionary
            config_dict = {
                'app_name': config.app_name,
                'version': config.version,
                'environment': config.environment,
                'debug': config.debug,
                'data': {
                    'raw_data_path': config.data.raw_data_path,
                    'processed_data_path': config.data.processed_data_path,
                    'chunk_size': config.data.chunk_size,
                    'coordinate_bounds': config.data.coordinate_bounds,
                    'outlier_threshold': config.data.outlier_threshold,
                    'min_call_frequency': config.data.min_call_frequency
                },
                'model': {
                    'test_size': config.model.test_size,
                    'random_state': config.model.random_state,
                    'cv_folds': config.model.cv_folds,
                    'n_jobs': config.model.n_jobs,
                    'optimize_hyperparameters': config.model.optimize_hyperparameters,
                    'save_models': config.model.save_models,
                    'model_output_path': config.model.model_output_path,
                    'xgb_params': config.model.xgb_params,
                    'rf_params': config.model.rf_params
                },
                'dashboard': {
                    'host': config.dashboard.host,
                    'port': config.dashboard.port,
                    'debug': config.dashboard.debug,
                    'cache_ttl': config.dashboard.cache_ttl,
                    'max_map_points': config.dashboard.max_map_points,
                    'default_date_range_days': config.dashboard.default_date_range_days
                },
                'logging': {
                    'level': config.logging.level,
                    'format': config.logging.format,
                    'log_file': config.logging.log_file,
                    'rotation': config.logging.rotation,
                    'retention': config.logging.retention
                },
                'database': {
                    'host': config.database.host,
                    'port': config.database.port,
                    'database': config.database.database,
                    'username': config.database.username,
                    'password': config.database.password,
                    'pool_size': config.database.pool_size
                },
                'api': {
                    'host': config.api.host,
                    'port': config.api.port,
                    'workers': config.api.workers,
                    'reload': config.api.reload,
                    'cors_origins': config.api.cors_origins
                }
            }
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save to YAML file
            with open(output_path, 'w') as file:
                yaml.dump(config_dict, file, default_flow_style=False, indent=2)
                
            logging.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
            raise


# Global configuration manager instance
_config_manager = None
_config = None


def get_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Get application configuration
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        AppConfig object
    """
    global _config_manager, _config
    
    if _config is None:
        if _config_manager is None:
            _config_manager = ConfigManager(config_path)
        _config = _config_manager.load_config()
    
    return _config


def reload_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Reload configuration from file
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        AppConfig object
    """
    global _config_manager, _config
    
    _config_manager = ConfigManager(config_path)
    _config = _config_manager.load_config()
    
    return _config 