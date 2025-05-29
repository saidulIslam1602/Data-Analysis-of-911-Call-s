"""
Unit Tests for Data Processing Module

This module contains comprehensive tests for the EmergencyCallsProcessor class
to ensure data quality, validation, and feature engineering work correctly.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from unittest.mock import Mock, patch

# Import the module to test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.processor import EmergencyCallsProcessor, DataValidator


class TestDataValidator:
    """Test the Pydantic data validator"""
    
    def test_valid_data(self):
        """Test validation with valid data"""
        valid_data = {
            'lat': 40.2677,
            'lng': -75.2797,
            'desc': 'Test description',
            'title': 'EMS: Test Emergency',
            'timeStamp': '2023-01-01 12:00:00',
            'addr': '123 Test St'
        }
        
        validator = DataValidator(**valid_data)
        assert validator.lat == 40.2677
        assert validator.lng == -75.2797
        assert validator.desc == 'Test description'
    
    def test_invalid_coordinates(self):
        """Test validation with invalid coordinates"""
        invalid_data = {
            'lat': 'invalid',
            'lng': -75.2797,
            'desc': 'Test description',
            'title': 'EMS: Test Emergency',
            'timeStamp': '2023-01-01 12:00:00',
            'addr': '123 Test St'
        }
        
        with pytest.raises(Exception):
            DataValidator(**invalid_data)


class TestEmergencyCallsProcessor:
    """Test the main data processor class"""
    
    @pytest.fixture
    def processor(self):
        """Create a processor instance for testing"""
        return EmergencyCallsProcessor()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample 911 calls data for testing"""
        data = {
            'lat': [40.2677, 40.2800, 40.2500, 40.2900],
            'lng': [-75.2797, -75.2900, -75.2600, -75.3000],
            'desc': [
                'Emergency at location 1',
                'Emergency at location 2',
                'Emergency at location 3',
                'Emergency at location 4'
            ],
            'zip': [19401, 19401, 19402, 19403],
            'title': [
                'EMS: CARDIAC EMERGENCY',
                'Fire: BUILDING FIRE',
                'Traffic: VEHICLE ACCIDENT',
                'EMS: BREATHING DIFFICULTY'
            ],
            'timeStamp': [
                '2023-01-01 12:00:00',
                '2023-01-01 13:30:00',
                '2023-01-02 09:15:00',
                '2023-01-02 16:45:00'
            ],
            'twp': ['Township A', 'Township B', 'Township A', 'Township C'],
            'addr': ['123 Main St', '456 Oak Ave', '789 Pine Rd', '321 Elm St'],
            'e': [1, 1, 1, 1]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_csv_file(self, sample_data):
        """Create a temporary CSV file with sample data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    def test_default_config(self, processor):
        """Test that default configuration is loaded correctly"""
        config = processor._default_config()
        
        assert isinstance(config, dict)
        assert 'outlier_threshold' in config
        assert 'coordinate_bounds' in config
        assert config['outlier_threshold'] == 3.0
        assert config['coordinate_bounds']['lat_min'] == 39.0
    
    def test_load_and_validate_data(self, processor, sample_csv_file):
        """Test loading and validation of data from CSV"""
        df = processor.load_and_validate_data(sample_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert 'lat' in df.columns
        assert 'lng' in df.columns
        assert 'timeStamp' in df.columns
    
    def test_load_invalid_file(self, processor):
        """Test loading from non-existent file"""
        with pytest.raises(FileNotFoundError):
            processor.load_and_validate_data('non_existent_file.csv')
    
    def test_validate_basic_structure_success(self, processor, sample_data):
        """Test successful basic structure validation"""
        # Should not raise an exception
        processor._validate_basic_structure(sample_data)
    
    def test_validate_basic_structure_missing_columns(self, processor):
        """Test basic structure validation with missing columns"""
        invalid_data = pd.DataFrame({'lat': [40.0], 'lng': [-75.0]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            processor._validate_basic_structure(invalid_data)
    
    def test_generate_quality_report(self, processor, sample_data):
        """Test data quality report generation"""
        report = processor._generate_quality_report(sample_data)
        
        assert isinstance(report, dict)
        assert 'total_records' in report
        assert 'missing_values' in report
        assert 'overall_score' in report
        assert report['total_records'] == 4
        assert isinstance(report['overall_score'], float)
        assert 0 <= report['overall_score'] <= 1
    
    def test_check_coordinate_validity(self, processor, sample_data):
        """Test coordinate validity checking"""
        result = processor._check_coordinate_validity(sample_data)
        
        assert isinstance(result, dict)
        assert 'valid_count' in result
        assert 'invalid_count' in result
        assert 'valid_ratio' in result
        assert result['valid_count'] + result['invalid_count'] == len(sample_data)
    
    def test_check_timestamp_validity(self, processor, sample_data):
        """Test timestamp validity checking"""
        result = processor._check_timestamp_validity(sample_data)
        
        assert isinstance(result, dict)
        assert 'valid_count' in result
        assert 'invalid_count' in result
        assert 'valid_ratio' in result
    
    def test_clean_data(self, processor, sample_data):
        """Test comprehensive data cleaning"""
        df_clean = processor.clean_data(sample_data)
        
        assert isinstance(df_clean, pd.DataFrame)
        assert len(df_clean) <= len(sample_data)  # Should be same or fewer records
        
        # Check that timestamps are converted to datetime
        if 'timeStamp' in df_clean.columns:
            assert pd.api.types.is_datetime64_any_dtype(df_clean['timeStamp'])
    
    def test_clean_timestamps(self, processor, sample_data):
        """Test timestamp cleaning"""
        df_clean = processor._clean_timestamps(sample_data)
        
        assert 'timeStamp' in df_clean.columns
        assert pd.api.types.is_datetime64_any_dtype(df_clean['timeStamp'])
    
    def test_clean_coordinates(self, processor, sample_data):
        """Test coordinate cleaning"""
        df_clean = processor._clean_coordinates(sample_data)
        
        # All coordinates in sample data are valid, so should have same length
        assert len(df_clean) == len(sample_data)
        
        # Test with invalid coordinates
        invalid_data = sample_data.copy()
        invalid_data.loc[0, 'lat'] = 999  # Invalid latitude
        
        df_clean_invalid = processor._clean_coordinates(invalid_data)
        assert len(df_clean_invalid) < len(invalid_data)
    
    def test_clean_text_fields(self, processor, sample_data):
        """Test text field cleaning"""
        # Add some messy text data
        messy_data = sample_data.copy()
        messy_data.loc[0, 'desc'] = '  Emergency with extra spaces!@#  '
        messy_data.loc[1, 'addr'] = 'address with strange chars &*%'
        
        df_clean = processor._clean_text_fields(messy_data)
        
        # Check that text is cleaned (uppercase, trimmed)
        assert df_clean.loc[0, 'desc'].startswith('EMERGENCY')
        assert not df_clean.loc[0, 'desc'].startswith('  ')
    
    def test_remove_outliers(self, processor, sample_data):
        """Test outlier removal"""
        # Add outlier data
        outlier_data = sample_data.copy()
        outlier_data.loc[len(outlier_data)] = {
            'lat': 50.0,  # Far outside normal range
            'lng': -80.0,  # Far outside normal range
            'desc': 'Outlier',
            'zip': 99999,
            'title': 'Test',
            'timeStamp': '2023-01-01 12:00:00',
            'twp': 'Test',
            'addr': 'Test',
            'e': 1
        }
        
        df_clean = processor._remove_outliers(outlier_data)
        
        # Should remove the outlier
        assert len(df_clean) < len(outlier_data)
    
    def test_engineer_features(self, processor, sample_data):
        """Test feature engineering"""
        # First clean the data to ensure proper datetime conversion
        df_clean = processor.clean_data(sample_data)
        df_features = processor.engineer_features(df_clean)
        
        assert isinstance(df_features, pd.DataFrame)
        assert len(df_features) == len(df_clean)
        
        # Check that new features are created
        assert 'emergency_category' in df_features.columns
        
        # Check temporal features if timestamp was properly processed
        if pd.api.types.is_datetime64_any_dtype(df_features['timeStamp']):
            assert 'year' in df_features.columns
            assert 'month' in df_features.columns
            assert 'hour' in df_features.columns
    
    def test_create_temporal_features(self, processor, sample_data):
        """Test temporal feature creation"""
        # Ensure timestamp is datetime
        sample_data['timeStamp'] = pd.to_datetime(sample_data['timeStamp'])
        
        df_temporal = processor._create_temporal_features(sample_data)
        
        # Check that temporal features are created
        expected_features = ['year', 'month', 'day', 'hour', 'dayofweek', 'season']
        for feature in expected_features:
            assert feature in df_temporal.columns
        
        # Check cyclical encoding
        assert 'hour_sin' in df_temporal.columns
        assert 'hour_cos' in df_temporal.columns
    
    def test_create_geospatial_features(self, processor, sample_data):
        """Test geospatial feature creation"""
        df_geo = processor._create_geospatial_features(sample_data)
        
        # Check that geospatial features are created
        assert 'geo_cluster' in df_geo.columns
        assert 'distance_from_center' in df_geo.columns
        assert 'lat_rounded' in df_geo.columns
        assert 'lng_rounded' in df_geo.columns
    
    def test_create_text_features(self, processor, sample_data):
        """Test text feature creation"""
        df_text = processor._create_text_features(sample_data)
        
        # Check that text features are created
        assert 'desc_length' in df_text.columns
        assert 'addr_length' in df_text.columns
        assert 'desc_word_count' in df_text.columns
        assert 'has_intersection' in df_text.columns
        assert 'has_apartment' in df_text.columns
    
    def test_create_statistical_features(self, processor, sample_data):
        """Test statistical feature creation"""
        # Add required features first
        sample_data['lat_rounded'] = sample_data['lat'].round(3)
        sample_data['lng_rounded'] = sample_data['lng'].round(3)
        sample_data['emergency_category'] = sample_data['title'].str.split(':').str[0]
        
        df_stats = processor._create_statistical_features(sample_data)
        
        # Check that statistical features are created
        assert 'location_call_frequency' in df_stats.columns
        assert 'twp_call_frequency' in df_stats.columns
        assert 'category_frequency' in df_stats.columns
    
    def test_prepare_for_modeling(self, processor, sample_data):
        """Test data preparation for modeling"""
        # First engineer features
        df_clean = processor.clean_data(sample_data)
        df_features = processor.engineer_features(df_clean)
        
        df_model, preprocessing_info = processor.prepare_for_modeling(df_features)
        
        assert isinstance(df_model, pd.DataFrame)
        assert isinstance(preprocessing_info, dict)
        assert 'categorical_columns' in preprocessing_info
        assert 'numerical_columns' in preprocessing_info
        assert 'label_encoders' in preprocessing_info
    
    def test_save_processed_data(self, processor, sample_data):
        """Test saving processed data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_data.parquet')
            
            processor.save_processed_data(sample_data, output_path)
            
            # Check that files are created
            assert os.path.exists(output_path)
            assert os.path.exists(output_path.replace('.parquet', '.csv'))
            
            # Check that data can be loaded back
            df_loaded = pd.read_parquet(output_path)
            assert len(df_loaded) == len(sample_data)


class TestEmergencyCallsProcessorIntegration:
    """Integration tests for the complete data processing pipeline"""
    
    def test_full_pipeline(self):
        """Test the complete data processing pipeline"""
        # Create sample data
        data = {
            'lat': [40.2677, 40.2800, 40.2500] * 10,  # Larger dataset
            'lng': [-75.2797, -75.2900, -75.2600] * 10,
            'desc': ['Emergency description'] * 30,
            'zip': [19401, 19402, 19403] * 10,
            'title': ['EMS: CARDIAC', 'Fire: BUILDING', 'Traffic: ACCIDENT'] * 10,
            'timeStamp': ['2023-01-01 12:00:00'] * 30,
            'twp': ['Township A', 'Township B', 'Township C'] * 10,
            'addr': ['123 Main St'] * 30,
            'e': [1] * 30
        }
        df = pd.DataFrame(data)
        
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            # Run full pipeline
            processor = EmergencyCallsProcessor()
            
            # Load and validate
            df_raw = processor.load_and_validate_data(csv_path)
            assert len(df_raw) == 30
            
            # Clean data
            df_clean = processor.clean_data(df_raw)
            assert len(df_clean) <= 30
            
            # Engineer features
            df_features = processor.engineer_features(df_clean)
            assert df_features.shape[1] > df_clean.shape[1]  # More columns after feature engineering
            
            # Prepare for modeling
            df_model, preprocessing_info = processor.prepare_for_modeling(df_features)
            assert isinstance(df_model, pd.DataFrame)
            assert isinstance(preprocessing_info, dict)
            
        finally:
            os.unlink(csv_path)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"]) 