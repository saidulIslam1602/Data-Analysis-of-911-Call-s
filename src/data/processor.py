"""
Advanced Data Processing Module for 911 Emergency Calls Analysis

This module provides comprehensive data processing capabilities including:
- Data validation and quality checks
- Feature engineering and extraction
- Time series data preparation
- Geospatial data processing
- Statistical data transformations
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict, List, Optional
from datetime import datetime, timedelta
import warnings
from loguru import logger
from pydantic import BaseModel, ValidationError
import geopandas as gpd
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans


class DataValidator(BaseModel):
    """Pydantic model for data validation"""
    lat: float
    lng: float
    desc: str
    title: str
    timeStamp: str
    addr: str


class EmergencyCallsProcessor:
    """Advanced processor for 911 emergency calls data"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the processor with configuration
        
        Args:
            config: Configuration dictionary for processing parameters
        """
        self.config = config or self._default_config()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.processed_data = None
        self.feature_columns = []
        
    def _default_config(self) -> Dict:
        """Default configuration for data processing"""
        return {
            'outlier_threshold': 3.0,
            'min_call_frequency': 5,
            'time_zones': ['US/Eastern'],
            'coordinate_bounds': {
                'lat_min': 39.0, 'lat_max': 42.0,
                'lng_min': -76.0, 'lng_max': -74.0
            }
        }
    
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and validate the raw data
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Validated DataFrame
        """
        logger.info(f"Loading data from {file_path}")
        
        try:
            # Load data with optimized dtypes
            dtypes = {
                'lat': 'float32',
                'lng': 'float32',
                'desc': 'string',
                'zip': 'Int64',
                'title': 'string',
                'twp': 'string',
                'addr': 'string',
                'e': 'int8'
            }
            
            df = pd.read_csv(file_path, dtype=dtypes)
            logger.info(f"Loaded {len(df):,} records")
            
            # Basic validation
            self._validate_basic_structure(df)
            
            # Data quality checks
            quality_report = self._generate_quality_report(df)
            logger.info(f"Data quality score: {quality_report['overall_score']:.2f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _validate_basic_structure(self, df: pd.DataFrame) -> None:
        """Validate basic data structure"""
        required_columns = ['lat', 'lng', 'desc', 'title', 'timeStamp', 'addr']
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate coordinate bounds
        bounds = self.config['coordinate_bounds']
        invalid_coords = (
            (df['lat'] < bounds['lat_min']) | (df['lat'] > bounds['lat_max']) |
            (df['lng'] < bounds['lng_min']) | (df['lng'] > bounds['lng_max'])
        )
        
        if invalid_coords.sum() > 0:
            logger.warning(f"Found {invalid_coords.sum()} records with invalid coordinates")
    
    def _generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive data quality report"""
        report = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_records': df.duplicated().sum(),
            'coordinate_validity': self._check_coordinate_validity(df),
            'timestamp_validity': self._check_timestamp_validity(df),
        }
        
        # Calculate overall quality score
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        duplicate_ratio = report['duplicate_records'] / len(df)
        coord_validity = report['coordinate_validity']['valid_ratio']
        time_validity = report['timestamp_validity']['valid_ratio']
        
        report['overall_score'] = (
            (1 - missing_ratio) * 0.3 +
            (1 - duplicate_ratio) * 0.2 +
            coord_validity * 0.25 +
            time_validity * 0.25
        )
        
        return report
    
    def _check_coordinate_validity(self, df: pd.DataFrame) -> Dict:
        """Check validity of coordinate data"""
        bounds = self.config['coordinate_bounds']
        valid_coords = (
            (df['lat'].between(bounds['lat_min'], bounds['lat_max'])) &
            (df['lng'].between(bounds['lng_min'], bounds['lng_max'])) &
            df['lat'].notna() & df['lng'].notna()
        )
        
        return {
            'valid_count': valid_coords.sum(),
            'invalid_count': (~valid_coords).sum(),
            'valid_ratio': valid_coords.mean()
        }
    
    def _check_timestamp_validity(self, df: pd.DataFrame) -> Dict:
        """Check validity of timestamp data"""
        try:
            timestamps = pd.to_datetime(df['timeStamp'], errors='coerce')
            valid_timestamps = timestamps.notna()
            
            return {
                'valid_count': valid_timestamps.sum(),
                'invalid_count': (~valid_timestamps).sum(),
                'valid_ratio': valid_timestamps.mean()
            }
        except:
            return {'valid_count': 0, 'invalid_count': len(df), 'valid_ratio': 0.0}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data cleaning
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning process")
        
        df_clean = df.copy()
        
        # Remove duplicates
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {initial_count - len(df_clean)} duplicate records")
        
        # Clean timestamps
        df_clean = self._clean_timestamps(df_clean)
        
        # Clean coordinates
        df_clean = self._clean_coordinates(df_clean)
        
        # Clean text fields
        df_clean = self._clean_text_fields(df_clean)
        
        # Remove outliers
        df_clean = self._remove_outliers(df_clean)
        
        logger.info(f"Data cleaning completed. Final records: {len(df_clean):,}")
        return df_clean
    
    def _clean_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate timestamps"""
        df_clean = df.copy()
        
        # Convert to datetime
        df_clean['timeStamp'] = pd.to_datetime(df_clean['timeStamp'], errors='coerce')
        
        # Remove records with invalid timestamps
        valid_timestamps = df_clean['timeStamp'].notna()
        df_clean = df_clean[valid_timestamps]
        
        # Set timezone if not present - handle DST ambiguity
        if df_clean['timeStamp'].dt.tz is None:
            try:
                df_clean['timeStamp'] = df_clean['timeStamp'].dt.tz_localize('US/Eastern', ambiguous='NaT')
                # Remove any NaT values created by ambiguous times
                df_clean = df_clean[df_clean['timeStamp'].notna()]
            except Exception as e:
                logger.warning(f"Timezone localization failed: {e}. Using UTC instead.")
                df_clean['timeStamp'] = df_clean['timeStamp'].dt.tz_localize('UTC')
        
        return df_clean
    
    def _clean_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate coordinates"""
        df_clean = df.copy()
        bounds = self.config['coordinate_bounds']
        
        # Remove records with invalid coordinates
        valid_coords = (
            df_clean['lat'].between(bounds['lat_min'], bounds['lat_max']) &
            df_clean['lng'].between(bounds['lng_min'], bounds['lng_max']) &
            df_clean['lat'].notna() & df_clean['lng'].notna()
        )
        
        df_clean = df_clean[valid_coords]
        return df_clean
    
    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text fields"""
        df_clean = df.copy()
        
        text_columns = ['desc', 'title', 'addr', 'twp']
        
        for col in text_columns:
            if col in df_clean.columns:
                # Remove extra whitespace
                df_clean[col] = df_clean[col].astype(str).str.strip()
                
                # Standardize case
                df_clean[col] = df_clean[col].str.upper()
                
                # Remove special characters (keep basic punctuation)
                df_clean[col] = df_clean[col].str.replace(r'[^\w\s\-&:/.]', '', regex=True)
        
        return df_clean
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers"""
        df_clean = df.copy()
        
        # Remove coordinate outliers using IQR method
        for col in ['lat', 'lng']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            df_clean = df_clean[~outliers]
        
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering")
        
        df_features = df.copy()
        
        # Extract emergency category
        df_features['emergency_category'] = df_features['title'].str.split(':').str[0]
        
        # Time-based features
        df_features = self._create_temporal_features(df_features)
        
        # Geospatial features
        df_features = self._create_geospatial_features(df_features)
        
        # Text analysis features
        df_features = self._create_text_features(df_features)
        
        # Statistical features
        df_features = self._create_statistical_features(df_features)
        
        # Update feature columns list
        self.feature_columns = [col for col in df_features.columns if col not in df.columns]
        
        logger.info(f"Created {len(self.feature_columns)} new features")
        return df_features
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df_temp = df.copy()
        
        # Basic time components
        df_temp['year'] = df_temp['timeStamp'].dt.year
        df_temp['month'] = df_temp['timeStamp'].dt.month
        df_temp['day'] = df_temp['timeStamp'].dt.day
        df_temp['hour'] = df_temp['timeStamp'].dt.hour
        df_temp['minute'] = df_temp['timeStamp'].dt.minute
        df_temp['dayofweek'] = df_temp['timeStamp'].dt.dayofweek
        df_temp['dayofyear'] = df_temp['timeStamp'].dt.dayofyear
        
        # Categorical time features
        df_temp['season'] = df_temp['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        df_temp['time_of_day'] = pd.cut(
            df_temp['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )
        
        df_temp['is_weekend'] = df_temp['dayofweek'].isin([5, 6]).astype(int)
        df_temp['is_business_hours'] = df_temp['hour'].between(9, 17).astype(int)
        
        # Cyclical encoding for time features
        df_temp['hour_sin'] = np.sin(2 * np.pi * df_temp['hour'] / 24)
        df_temp['hour_cos'] = np.cos(2 * np.pi * df_temp['hour'] / 24)
        df_temp['month_sin'] = np.sin(2 * np.pi * df_temp['month'] / 12)
        df_temp['month_cos'] = np.cos(2 * np.pi * df_temp['month'] / 12)
        df_temp['dayofweek_sin'] = np.sin(2 * np.pi * df_temp['dayofweek'] / 7)
        df_temp['dayofweek_cos'] = np.cos(2 * np.pi * df_temp['dayofweek'] / 7)
        
        return df_temp
    
    def _create_geospatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create geospatial features"""
        df_geo = df.copy()
        
        # Create geographic clusters
        coords = df_geo[['lat', 'lng']].values
        kmeans = KMeans(n_clusters=20, random_state=42)
        df_geo['geo_cluster'] = kmeans.fit_predict(coords)
        
        # Distance from city center (approximate)
        city_center_lat, city_center_lng = 40.2677, -75.2797  # Montgomery County center
        df_geo['distance_from_center'] = np.sqrt(
            (df_geo['lat'] - city_center_lat) ** 2 + 
            (df_geo['lng'] - city_center_lng) ** 2
        )
        
        # Coordinate-based features
        df_geo['lat_rounded'] = df_geo['lat'].round(3)
        df_geo['lng_rounded'] = df_geo['lng'].round(3)
        
        return df_geo
    
    def _create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create text-based features"""
        df_text = df.copy()
        
        # Description length
        df_text['desc_length'] = df_text['desc'].str.len()
        df_text['addr_length'] = df_text['addr'].str.len()
        
        # Word count in description
        df_text['desc_word_count'] = df_text['desc'].str.split().str.len()
        
        # Address features
        df_text['has_intersection'] = df_text['addr'].str.contains('&', na=False).astype(int)
        df_text['has_apartment'] = df_text['addr'].str.contains(r'APT|UNIT|#', na=False).astype(int)
        
        return df_text
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical aggregation features"""
        df_stats = df.copy()
        
        # Call frequency by location
        location_counts = df_stats.groupby(['lat_rounded', 'lng_rounded']).size()
        df_stats['location_call_frequency'] = df_stats.apply(
            lambda row: location_counts.get((row['lat_rounded'], row['lng_rounded']), 0), axis=1
        )
        
        # Call frequency by township
        if 'twp' in df_stats.columns:
            twp_counts = df_stats['twp'].value_counts()
            df_stats['twp_call_frequency'] = df_stats['twp'].map(twp_counts).fillna(0)
        
        # Emergency category frequency
        category_counts = df_stats['emergency_category'].value_counts()
        df_stats['category_frequency'] = df_stats['emergency_category'].map(category_counts)
        
        return df_stats
    
    def prepare_for_modeling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare data for machine learning models
        
        Args:
            df: Feature-engineered DataFrame
            
        Returns:
            Tuple of (processed DataFrame, preprocessing info)
        """
        logger.info("Preparing data for modeling")
        
        df_model = df.copy()
        preprocessing_info = {}
        
        # Encode categorical variables
        categorical_columns = df_model.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if col not in ['timeStamp']:  # Exclude timestamp
                le = LabelEncoder()
                df_model[f'{col}_encoded'] = le.fit_transform(df_model[col].astype(str))
                self.label_encoders[col] = le
        
        # Scale numerical features
        numerical_columns = df_model.select_dtypes(include=[np.number]).columns
        numerical_columns = [col for col in numerical_columns if col not in ['lat', 'lng']]  # Keep original coordinates
        
        df_model[numerical_columns] = self.scaler.fit_transform(df_model[numerical_columns])
        
        preprocessing_info = {
            'categorical_columns': list(categorical_columns),
            'numerical_columns': list(numerical_columns),
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }
        
        self.processed_data = df_model
        return df_model, preprocessing_info
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """Save processed data to file"""
        logger.info(f"Saving processed data to {output_path}")
        
        # Save as parquet for better performance
        df.to_parquet(output_path, index=False)
        
        # Also save as CSV for compatibility
        csv_path = output_path.replace('.parquet', '.csv')
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Data saved to {output_path} and {csv_path}")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process 911 emergency calls data')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output directory path')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = EmergencyCallsProcessor()
    
    # Process data
    df_raw = processor.load_and_validate_data(args.input)
    df_clean = processor.clean_data(df_raw)
    df_features = processor.engineer_features(df_clean)
    df_model, preprocessing_info = processor.prepare_for_modeling(df_features)
    
    # Save processed data
    import os
    os.makedirs(args.output, exist_ok=True)
    processor.save_processed_data(df_model, os.path.join(args.output, 'processed_data.parquet'))
    
    logger.info("Data processing completed successfully!")


if __name__ == "__main__":
    main() 