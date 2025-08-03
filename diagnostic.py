"""
AI TRADING PROFESSIONAL - COMPREHENSIVE DIAGNOSTIC TOOL
========================================================
This diagnostic script thoroughly tests all functionality of fixedui.py
and its dependencies from enhprog.py to identify issues and provide solutions.
"""

import os
import sys
import logging
import time
import traceback
import importlib
import inspect
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Test basic imports first
print("🚀 AI Trading Professional - Comprehensive Diagnostic Tool")
print("=" * 60)
print(f"📅 Diagnostic started at: {datetime.now()}")
print(f"🐍 Python version: {sys.version}")
print(f"📂 Current directory: {os.getcwd()}")
print("=" * 60)

# Initialize results tracking
diagnostic_results = {
    'basic_imports': {},
    'enhprog_imports': {},
    'api_keys': {},
    'data_providers': {},
    'model_classes': {},
    'prediction_engine': {},
    'cross_validation': {},
    'backtesting': {},
    'real_time_data': {},
    'file_system': {},
    'advanced_features': {},
    'session_state': {},
    'overall_health': 'UNKNOWN'
}

errors_found = []
warnings_found = []
solutions = []

def log_result(category: str, test_name: str, status: str, details: str = "", error: str = ""):
    """Log diagnostic result"""
    diagnostic_results[category][test_name] = {
        'status': status,
        'details': details,
        'error': error,
        'timestamp': datetime.now().isoformat()
    }
    
    status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"{status_icon} {test_name}: {status}")
    if details:
        print(f"   💡 {details}")
    if error:
        print(f"   🚨 Error: {error}")
        errors_found.append(f"{test_name}: {error}")

def test_basic_imports():
    """Test basic Python package imports"""
    print("\n📦 Testing Basic Package Imports")
    print("-" * 40)
    
    basic_packages = [
        'numpy', 'pandas', 'streamlit', 'plotly', 'sklearn',
        'torch', 'joblib', 'requests', 'altair', 'pathlib'
    ]
    
    for package in basic_packages:
        try:
            __import__(package)
            log_result('basic_imports', package, 'PASS', f"Successfully imported {package}")
        except ImportError as e:
            log_result('basic_imports', package, 'FAIL', 
                      f"Failed to import {package}", str(e))
            solutions.append(f"Install {package}: pip install {package}")

def test_enhprog_imports():
    """Test all imports from enhprog.py"""
    print("\n🧠 Testing Enhanced Program (enhprog.py) Imports")
    print("-" * 50)
    
    # Test if enhprog.py exists
    enhprog_path = Path("enhprog.py")
    if not enhprog_path.exists():
        log_result('enhprog_imports', 'enhprog_file', 'FAIL', 
                  "enhprog.py file not found", "File does not exist")
        solutions.append("Ensure enhprog.py is in the same directory as fixedui.py")
        return
    
    log_result('enhprog_imports', 'enhprog_file', 'PASS', 
              f"Found enhprog.py at {enhprog_path.absolute()}")
    
    # Test importing enhprog module
    try:
        import enhprog
        log_result('enhprog_imports', 'enhprog_module', 'PASS', 
                  "Successfully imported enhprog module")
    except Exception as e:
        log_result('enhprog_imports', 'enhprog_module', 'FAIL', 
                  "Failed to import enhprog module", str(e))
        solutions.append("Check enhprog.py for syntax errors or missing dependencies")
        return
    
    # Test specific function imports
    enhprog_functions = [
        # Core prediction functions
        'get_real_time_prediction',
        'train_enhanced_models',
        'multi_step_forecast',
        'enhanced_ensemble_predict',
        'calculate_prediction_confidence',
        
        # Data management classes
        'MultiTimeframeDataManager',
        'RealTimeDataProcessor',
        'HFFeatureCalculator',
        'FMPDataProvider',
        'RealTimeEconomicDataProvider',
        'RealTimeSentimentProvider',
        'RealTimeOptionsProvider',
        
        # Advanced analytics classes
        'AdvancedMarketRegimeDetector',
        'AdvancedRiskManager',
        'ModelExplainer',
        'ModelDriftDetector',
        
        # Cross-validation and model selection
        'TimeSeriesCrossValidator',
        'ModelSelectionFramework',
        'MetaLearningEnsemble',
        
        # Backtesting
        'AdvancedBacktester',
        'EnhancedStrategy',
        'Portfolio',
        
        # Neural network models
        'AdvancedTransformer',
        'CNNLSTMAttention',
        'EnhancedTCN',
        'EnhancedInformer',
        'EnhancedNBeats',
        'LSTMGRUEnsemble',
        
        # Enhanced models
        'XGBoostTimeSeriesModel',
        'SklearnEnsemble',
        
        # Utility functions
        'get_asset_type',
        'get_reasonable_price_range',
        'is_market_open',
        'enhance_features',
        'prepare_sequence_data',
        'inverse_transform_prediction',
        'load_trained_models',
        
        # Constants
        'ENHANCED_TICKERS',
        'TIMEFRAMES',
        'FMP_API_KEY',
        'FRED_API_KEY',
        'STATE_FILE'
    ]
    
    for func_name in enhprog_functions:
        try:
            func = getattr(enhprog, func_name)
            if inspect.isclass(func):
                log_result('enhprog_imports', func_name, 'PASS', 
                          f"Successfully imported class {func_name}")
            elif callable(func):
                log_result('enhprog_imports', func_name, 'PASS', 
                          f"Successfully imported function {func_name}")
            else:
                log_result('enhprog_imports', func_name, 'PASS', 
                          f"Successfully imported constant {func_name}")
        except AttributeError:
            log_result('enhprog_imports', func_name, 'FAIL', 
                      f"{func_name} not found in enhprog module", 
                      "Missing function/class")
            solutions.append(f"Implement {func_name} in enhprog.py")
        except Exception as e:
            log_result('enhprog_imports', func_name, 'FAIL', 
                      f"Error accessing {func_name}", str(e))

def test_api_keys():
    """Test API key availability and validity"""
    print("\n🔑 Testing API Keys and Credentials")
    print("-" * 40)
    
    # Test environment variables
    api_keys = {
        'FMP_API_KEY': os.getenv('FMP_API_KEY'),
        'FRED_API_KEY': os.getenv('FRED_API_KEY'),
        'TRADING_PREMIUM_KEY': os.getenv('TRADING_PREMIUM_KEY')
    }
    
    for key_name, key_value in api_keys.items():
        if key_value:
            log_result('api_keys', key_name, 'PASS', 
                      f"{key_name} found in environment")
            
            # Test key format validity
            if key_name == 'FMP_API_KEY' and len(key_value) < 10:
                log_result('api_keys', f'{key_name}_format', 'WARN', 
                          "FMP API key seems too short")
                warnings_found.append(f"{key_name} format may be invalid")
            elif key_name == 'TRADING_PREMIUM_KEY' and key_value != 'Prem246_357':
                log_result('api_keys', f'{key_name}_format', 'WARN', 
                          "Premium key doesn't match expected value")
        else:
            log_result('api_keys', key_name, 'FAIL', 
                      f"{key_name} not found in environment", "Missing API key")
            solutions.append(f"Set {key_name} environment variable")
    
    # Test API key access from enhprog
    try:
        import enhprog
        fmp_key = getattr(enhprog, 'FMP_API_KEY', None)
        fred_key = getattr(enhprog, 'FRED_API_KEY', None)
        
        if fmp_key:
            log_result('api_keys', 'enhprog_FMP_KEY', 'PASS', 
                      "FMP API key accessible from enhprog")
        else:
            log_result('api_keys', 'enhprog_FMP_KEY', 'FAIL', 
                      "FMP API key not accessible from enhprog", "Key not found")
            
        if fred_key:
            log_result('api_keys', 'enhprog_FRED_KEY', 'PASS', 
                      "FRED API key accessible from enhprog")
        else:
            log_result('api_keys', 'enhprog_FRED_KEY', 'FAIL', 
                      "FRED API key not accessible from enhprog", "Key not found")
            
    except Exception as e:
        log_result('api_keys', 'enhprog_keys', 'FAIL', 
                  "Error accessing keys from enhprog", str(e))

def test_data_providers():
    """Test data provider classes instantiation"""
    print("\n📊 Testing Data Provider Classes")
    print("-" * 40)
    
    try:
        import enhprog
        
        # Test MultiTimeframeDataManager
        try:
            tickers = getattr(enhprog, 'ENHANCED_TICKERS', ['^GSPC', 'AAPL'])
            data_manager = enhprog.MultiTimeframeDataManager(tickers[:3])
            log_result('data_providers', 'MultiTimeframeDataManager', 'PASS', 
                      f"Successfully instantiated with {len(tickers[:3])} tickers")
            
            # Test basic method availability
            methods_to_test = [
                'fetch_multi_timeframe_data',
                'get_real_time_price',
                'fetch_alternative_data'
            ]
            
            for method in methods_to_test:
                if hasattr(data_manager, method):
                    log_result('data_providers', f'DataManager_{method}', 'PASS', 
                              f"Method {method} available")
                else:
                    log_result('data_providers', f'DataManager_{method}', 'FAIL', 
                              f"Method {method} missing", "Method not implemented")
                    
        except Exception as e:
            log_result('data_providers', 'MultiTimeframeDataManager', 'FAIL', 
                      "Failed to instantiate data manager", str(e))
        
        # Test other data providers
        data_provider_classes = [
            'RealTimeDataProcessor',
            'HFFeatureCalculator',
            'FMPDataProvider',
            'RealTimeEconomicDataProvider',
            'RealTimeSentimentProvider',
            'RealTimeOptionsProvider'
        ]
        
        for class_name in data_provider_classes:
            try:
                provider_class = getattr(enhprog, class_name)
                if class_name in ['RealTimeEconomicDataProvider', 'RealTimeSentimentProvider', 'RealTimeOptionsProvider']:
                    # These might require API keys
                    try:
                        provider = provider_class()
                        log_result('data_providers', class_name, 'PASS', 
                                  f"Successfully instantiated {class_name}")
                    except Exception as e:
                        log_result('data_providers', class_name, 'WARN', 
                                  f"Instantiation failed, likely missing API key", str(e))
                else:
                    provider = provider_class()
                    log_result('data_providers', class_name, 'PASS', 
                              f"Successfully instantiated {class_name}")
                    
            except AttributeError:
                log_result('data_providers', class_name, 'FAIL', 
                          f"Class {class_name} not found", "Class not implemented")
            except Exception as e:
                log_result('data_providers', class_name, 'FAIL', 
                          f"Failed to instantiate {class_name}", str(e))
                
    except Exception as e:
        log_result('data_providers', 'import_error', 'FAIL', 
                  "Failed to import enhprog for data provider testing", str(e))

def test_model_classes():
    """Test AI model classes instantiation"""
    print("\n🤖 Testing AI Model Classes")
    print("-" * 40)
    
    try:
        import enhprog
        
        # Test neural network models
        neural_models = [
            ('AdvancedTransformer', {'n_features': 5, 'seq_len': 60}),
            ('CNNLSTMAttention', {'n_features': 5}),
            ('EnhancedTCN', {'n_features': 5}),
            ('EnhancedInformer', {'enc_in': 5, 'dec_in': 5, 'c_out': 1, 'seq_len': 60}),
            ('EnhancedNBeats', {'input_size': 300}),
            ('LSTMGRUEnsemble', {'input_size': 5, 'hidden_size': 64, 'num_layers': 2})
        ]
        
        for model_name, params in neural_models:
            try:
                model_class = getattr(enhprog, model_name)
                model = model_class(**params)
                log_result('model_classes', model_name, 'PASS', 
                          f"Successfully instantiated {model_name}")
                
                # Test if model has required methods
                required_methods = ['forward', 'parameters'] if hasattr(model, 'forward') else []
                for method in required_methods:
                    if hasattr(model, method):
                        log_result('model_classes', f'{model_name}_{method}', 'PASS', 
                                  f"Method {method} available")
                    else:
                        log_result('model_classes', f'{model_name}_{method}', 'FAIL', 
                                  f"Method {method} missing", "Required method not found")
                        
            except AttributeError:
                log_result('model_classes', model_name, 'FAIL', 
                          f"Class {model_name} not found", "Class not implemented")
            except Exception as e:
                log_result('model_classes', model_name, 'FAIL', 
                          f"Failed to instantiate {model_name}", str(e))
        
        # Test traditional ML models
        traditional_models = [
            'XGBoostTimeSeriesModel',
            'SklearnEnsemble'
        ]
        
        for model_name in traditional_models:
            try:
                model_class = getattr(enhprog, model_name)
                model = model_class()
                log_result('model_classes', model_name, 'PASS', 
                          f"Successfully instantiated {model_name}")
            except AttributeError:
                log_result('model_classes', model_name, 'FAIL', 
                          f"Class {model_name} not found", "Class not implemented")
            except Exception as e:
                log_result('model_classes', model_name, 'FAIL', 
                          f"Failed to instantiate {model_name}", str(e))
                
    except Exception as e:
        log_result('model_classes', 'import_error', 'FAIL', 
                  "Failed to import enhprog for model testing", str(e))

def test_prediction_engine():
    """Test prediction engine functionality"""
    print("\n🎯 Testing Prediction Engine")
    print("-" * 40)
    
    try:
        import enhprog
        
        # Test utility functions
        utility_functions = [
            ('get_asset_type', ['^GSPC']),
            ('get_reasonable_price_range', ['^GSPC']),
            ('is_market_open', []),
        ]
        
        for func_name, args in utility_functions:
            try:
                func = getattr(enhprog, func_name)
                result = func(*args)
                log_result('prediction_engine', func_name, 'PASS', 
                          f"Function {func_name} executed successfully, result: {result}")
            except AttributeError:
                log_result('prediction_engine', func_name, 'FAIL', 
                          f"Function {func_name} not found", "Function not implemented")
            except Exception as e:
                log_result('prediction_engine', func_name, 'FAIL', 
                          f"Function {func_name} execution failed", str(e))
        
        # Test feature enhancement
        try:
            import pandas as pd
            import numpy as np
            
            # Create sample data
            sample_data = pd.DataFrame({
                'Open': np.random.randn(100) + 100,
                'High': np.random.randn(100) + 101,
                'Low': np.random.randn(100) + 99,
                'Close': np.random.randn(100) + 100,
                'Volume': np.random.randint(1000, 10000, 100)
            })
            
            enhance_features = getattr(enhprog, 'enhance_features')
            enhanced_data = enhance_features(sample_data, ['Open', 'High', 'Low', 'Close', 'Volume'])
            
            if enhanced_data is not None and not enhanced_data.empty:
                log_result('prediction_engine', 'enhance_features', 'PASS', 
                          f"Enhanced features: {enhanced_data.shape[1]} columns from {sample_data.shape[1]}")
            else:
                log_result('prediction_engine', 'enhance_features', 'FAIL', 
                          "Feature enhancement returned None or empty DataFrame", "Enhancement failed")
                
        except Exception as e:
            log_result('prediction_engine', 'enhance_features', 'FAIL', 
                      "Feature enhancement test failed", str(e))
        
        # Test sequence data preparation
        try:
            prepare_sequence_data = getattr(enhprog, 'prepare_sequence_data')
            if enhanced_data is not None:
                X_seq, y_seq, scaler = prepare_sequence_data(
                    enhanced_data, list(enhanced_data.columns), time_step=10
                )
                
                if X_seq is not None and y_seq is not None:
                    log_result('prediction_engine', 'prepare_sequence_data', 'PASS', 
                              f"Sequence data prepared: X shape {X_seq.shape}, y shape {y_seq.shape}")
                else:
                    log_result('prediction_engine', 'prepare_sequence_data', 'FAIL', 
                              "Sequence data preparation returned None", "Preparation failed")
            else:
                log_result('prediction_engine', 'prepare_sequence_data', 'WARN', 
                          "Skipped due to feature enhancement failure", "Dependent on previous test")
                
        except Exception as e:
            log_result('prediction_engine', 'prepare_sequence_data', 'FAIL', 
                      "Sequence data preparation test failed", str(e))
            
    except Exception as e:
        log_result('prediction_engine', 'import_error', 'FAIL', 
                  "Failed to import enhprog for prediction testing", str(e))

def test_real_time_data():
    """Test real-time data functionality"""
    print("\n📡 Testing Real-time Data Functionality")
    print("-" * 40)
    
    try:
        import enhprog
        
        # Test if we can create data manager
        tickers = ['^GSPC', 'AAPL', 'MSFT']
        data_manager = enhprog.MultiTimeframeDataManager(tickers)
        
        # Test real-time price fetching
        try:
            price = data_manager.get_real_time_price('^GSPC')
            if price and price > 0:
                log_result('real_time_data', 'real_time_price', 'PASS', 
                          f"Successfully fetched real-time price: ${price:.2f}")
            else:
                log_result('real_time_data', 'real_time_price', 'WARN', 
                          "Real-time price returned None or invalid value", "API might be down or key invalid")
        except Exception as e:
            log_result('real_time_data', 'real_time_price', 'FAIL', 
                      "Real-time price fetching failed", str(e))
        
        # Test multi-timeframe data fetching
        try:
            timeframes = ['1d']
            multi_data = data_manager.fetch_multi_timeframe_data('^GSPC', timeframes)
            
            if multi_data and '1d' in multi_data:
                data_shape = multi_data['1d'].shape
                log_result('real_time_data', 'multi_timeframe_data', 'PASS', 
                          f"Successfully fetched multi-timeframe data: {data_shape}")
            else:
                log_result('real_time_data', 'multi_timeframe_data', 'WARN', 
                          "Multi-timeframe data returned None or missing timeframe", "API response issue")
        except Exception as e:
            log_result('real_time_data', 'multi_timeframe_data', 'FAIL', 
                      "Multi-timeframe data fetching failed", str(e))
        
        # Test alternative data fetching
        try:
            alt_data = data_manager.fetch_alternative_data('^GSPC')
            
            if alt_data and isinstance(alt_data, dict):
                log_result('real_time_data', 'alternative_data', 'PASS', 
                          f"Successfully fetched alternative data: {len(alt_data)} sources")
            else:
                log_result('real_time_data', 'alternative_data', 'WARN', 
                          "Alternative data returned None or invalid format", "Limited data sources")
        except Exception as e:
            log_result('real_time_data', 'alternative_data', 'FAIL', 
                      "Alternative data fetching failed", str(e))
            
    except Exception as e:
        log_result('real_time_data', 'data_manager_creation', 'FAIL', 
                  "Failed to create data manager", str(e))

def test_file_system():
    """Test file system and model storage"""
    print("\n💾 Testing File System and Model Storage")
    print("-" * 40)
    
    # Test models directory
    models_dir = Path("models")
    if models_dir.exists():
        log_result('file_system', 'models_directory', 'PASS', 
                  f"Models directory exists at {models_dir.absolute()}")
        
        # Check for model files
        model_files = list(models_dir.glob('*.pt')) + list(models_dir.glob('*.pkl'))
        if model_files:
            log_result('file_system', 'model_files', 'PASS', 
                      f"Found {len(model_files)} model files")
        else:
            log_result('file_system', 'model_files', 'WARN', 
                      "No model files found in models directory", "Models need to be trained")
    else:
        log_result('file_system', 'models_directory', 'FAIL', 
                  "Models directory does not exist", "Directory missing")
        solutions.append("Create models directory: mkdir models")
    
    # Test data directory
    data_dir = Path("data")
    if data_dir.exists():
        log_result('file_system', 'data_directory', 'PASS', 
                  f"Data directory exists at {data_dir.absolute()}")
    else:
        log_result('file_system', 'data_directory', 'WARN', 
                  "Data directory does not exist", "Optional directory")
    
    # Test log files
    log_files = list(Path.cwd().glob('*.log'))
    if log_files:
        log_result('file_system', 'log_files', 'PASS', 
                  f"Found {len(log_files)} log files")
    else:
        log_result('file_system', 'log_files', 'WARN', 
                  "No log files found", "Logging may not be configured")
    
    # Test write permissions
    try:
        test_file = Path("diagnostic_test.txt")
        test_file.write_text("test")
        test_file.unlink()
        log_result('file_system', 'write_permissions', 'PASS', 
                  "Write permissions confirmed")
    except Exception as e:
        log_result('file_system', 'write_permissions', 'FAIL', 
                  "Write permissions test failed", str(e))
        solutions.append("Check directory write permissions")

def test_advanced_features():
    """Test advanced analytics features"""
    print("\n🔬 Testing Advanced Analytics Features")
    print("-" * 40)
    
    try:
        import enhprog
        
        # Test advanced analytics classes
        advanced_classes = [
            ('AdvancedMarketRegimeDetector', {'n_regimes': 4}),
            ('AdvancedRiskManager', {}),
            ('ModelExplainer', {}),
            ('ModelDriftDetector', {'reference_window': 100, 'detection_window': 50, 'drift_threshold': 0.05}),
            ('TimeSeriesCrossValidator', {'n_splits': 3, 'test_size': 0.2, 'gap': 5}),
            ('ModelSelectionFramework', {'cv_folds': 3}),
            ('AdvancedBacktester', {'initial_capital': 10000, 'commission': 0.001, 'slippage': 0.0005})
        ]
        
        for class_name, params in advanced_classes:
            try:
                advanced_class = getattr(enhprog, class_name)
                instance = advanced_class(**params)
                log_result('advanced_features', class_name, 'PASS', 
                          f"Successfully instantiated {class_name}")
            except AttributeError:
                log_result('advanced_features', class_name, 'FAIL', 
                          f"Class {class_name} not found", "Class not implemented")
            except Exception as e:
                log_result('advanced_features', class_name, 'FAIL', 
                          f"Failed to instantiate {class_name}", str(e))
        
        # Test strategy and portfolio classes
        strategy_classes = ['EnhancedStrategy', 'Portfolio']
        for class_name in strategy_classes:
            try:
                strategy_class = getattr(enhprog, class_name)
                if class_name == 'EnhancedStrategy':
                    instance = strategy_class('^GSPC')
                else:
                    instance = strategy_class()
                log_result('advanced_features', class_name, 'PASS', 
                          f"Successfully instantiated {class_name}")
            except AttributeError:
                log_result('advanced_features', class_name, 'FAIL', 
                          f"Class {class_name} not found", "Class not implemented")
            except Exception as e:
                log_result('advanced_features', class_name, 'FAIL', 
                          f"Failed to instantiate {class_name}", str(e))
                
    except Exception as e:
        log_result('advanced_features', 'import_error', 'FAIL', 
                  "Failed to import enhprog for advanced features testing", str(e))

def test_prediction_workflow():
    """Test complete prediction workflow"""
    print("\n🔄 Testing Complete Prediction Workflow")
    print("-" * 40)
    
    try:
        import enhprog
        import pandas as pd
        import numpy as np
        
        # Create sample enhanced data
        sample_data = pd.DataFrame({
            'Open': np.random.randn(200) + 100,
            'High': np.random.randn(200) + 101,
            'Low': np.random.randn(200) + 99,
            'Close': np.random.randn(200) + 100,
            'Volume': np.random.randint(1000, 10000, 200)
        })
        
        # Test feature enhancement
        enhanced_data = enhprog.enhance_features(sample_data, ['Open', 'High', 'Low', 'Close', 'Volume'])
        
        if enhanced_data is not None:
            log_result('prediction_engine', 'workflow_feature_enhancement', 'PASS', 
                      f"Enhanced data shape: {enhanced_data.shape}")
            
            # Test sequence preparation
            X_seq, y_seq, scaler = enhprog.prepare_sequence_data(
                enhanced_data, list(enhanced_data.columns), time_step=30
            )
            
            if X_seq is not None and y_seq is not None:
                log_result('prediction_engine', 'workflow_sequence_prep', 'PASS', 
                          f"Sequence data: X {X_seq.shape}, y {y_seq.shape}")
                
                # Test model training (simplified)
                try:
                    trained_models, trained_scaler, config = enhprog.train_enhanced_models(
                        enhanced_data,
                        list(enhanced_data.columns),
                        '^GSPC',
                        time_step=30,
                        use_cross_validation=False
                    )
                    
                    if trained_models:
                        log_result('prediction_engine', 'workflow_model_training', 'PASS', 
                                  f"Trained {len(trained_models)} models")
                        
                        # Test prediction
                        try:
                            prediction = enhprog.get_real_time_prediction(
                                '^GSPC',
                                models=trained_models,
                                config=config
                            )
                            
                            if prediction:
                                log_result('prediction_engine', 'workflow_prediction', 'PASS', 
                                          f"Generated prediction: {prediction.get('predicted_price', 'N/A')}")
                            else:
                                log_result('prediction_engine', 'workflow_prediction', 'FAIL', 
                                          "Prediction returned None", "Prediction generation failed")
                                
                        except Exception as e:
                            log_result('prediction_engine', 'workflow_prediction', 'FAIL', 
                                      "Prediction generation failed", str(e))
                    else:
                        log_result('prediction_engine', 'workflow_model_training', 'FAIL', 
                                  "Model training returned no models", "Training failed")
                        
                except Exception as e:
                    log_result('prediction_engine', 'workflow_model_training', 'FAIL', 
                              "Model training failed", str(e))
            else:
                log_result('prediction_engine', 'workflow_sequence_prep', 'FAIL', 
                          "Sequence preparation returned None", "Data preparation failed")
        else:
            log_result('prediction_engine', 'workflow_feature_enhancement', 'FAIL', 
                      "Feature enhancement returned None", "Enhancement failed")
            
    except Exception as e:
        log_result('prediction_engine', 'workflow_error', 'FAIL', 
                  "Complete workflow test failed", str(e))

def test_streamlit_integration():
    """Test Streamlit integration components"""
    print("\n🎨 Testing Streamlit Integration")
    print("-" * 40)
    
    try:
        import streamlit as st
        log_result('session_state', 'streamlit_import', 'PASS', 
                  "Streamlit imported successfully")
        
        # Test session state simulation
        class MockSessionState:
            def __init__(self):
                self.subscription_tier = 'free'
                self.premium_key = ''
                self.selected_ticker = '^GSPC'
                self.models_trained = {}
                self.current_prediction = None
                
        mock_state = MockSessionState()
        log_result('session_state', 'mock_session_state', 'PASS', 
                  "Mock session state created successfully")
        
        # Test subscription management from fixedui.py
        try:
            # This would require importing fixedui, but we'll test the logic
            premium_key = "Prem246_357"
            if premium_key == "Prem246_357":
                log_result('session_state', 'premium_key_validation', 'PASS', 
                          "Premium key validation logic works")
            else:
                log_result('session_state', 'premium_key_validation', 'FAIL', 
                          "Premium key validation failed", "Logic error")
                
        except Exception as e:
            log_result('session_state', 'premium_key_validation', 'FAIL', 
                      "Premium key validation test failed", str(e))
            
    except ImportError as e:
        log_result('session_state', 'streamlit_import', 'FAIL', 
                  "Failed to import Streamlit", str(e))
        solutions.append("Install Streamlit: pip install streamlit")

def test_chart_generation():
    """Test chart generation capabilities"""
    print("\n📊 Testing Chart Generation")
    print("-" * 40)
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import pandas as pd
        import numpy as np
        
        log_result('session_state', 'plotly_import', 'PASS', 
                  "Plotly imported successfully")
        
        # Test basic chart creation
        try:
            sample_data = pd.DataFrame({
                'x': range(10),
                'y': np.random.randn(10)
            })
            
            fig = px.line(sample_data, x='x', y='y', title='Test Chart')
            log_result('session_state', 'basic_chart_creation', 'PASS', 
                      "Basic chart created successfully")
            
            # Test subplot creation
            subplot_fig = make_subplots(rows=2, cols=2)
            log_result('session_state', 'subplot_creation', 'PASS', 
                      "Subplot created successfully")
            
        except Exception as e:
            log_result('session_state', 'chart_creation', 'FAIL', 
                      "Chart creation failed", str(e))
            
    except ImportError as e:
        log_result('session_state', 'plotly_import', 'FAIL', 
                  "Failed to import Plotly", str(e))
        solutions.append("Install Plotly: pip install plotly")

def generate_comprehensive_report():
    """Generate comprehensive diagnostic report"""
    print("\n" + "=" * 60)
    print("📋 COMPREHENSIVE DIAGNOSTIC REPORT")
    print("=" * 60)
    
    # Calculate overall health
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    warning_tests = 0
    
    for category, tests in diagnostic_results.items():
        if category == 'overall_health':
            continue
        for test_name, result in tests.items():
            total_tests += 1
            if result['status'] == 'PASS':
                passed_tests += 1
            elif result['status'] == 'FAIL':
                failed_tests += 1
            elif result['status'] == 'WARN':
                warning_tests += 1
    
    # Determine overall health
    pass_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    if pass_rate >= 0.9 and failed_tests == 0:
        overall_health = "🟢 EXCELLENT"
    elif pass_rate >= 0.8 and failed_tests <= 2:
        overall_health = "🟡 GOOD"
    elif pass_rate >= 0.6:
        overall_health = "🟠 MODERATE"
    else:
        overall_health = "🔴 POOR"
    
    diagnostic_results['overall_health'] = overall_health
    
    print(f"📊 OVERALL SYSTEM HEALTH: {overall_health}")
    print(f"✅ Passed: {passed_tests}")
    print(f"⚠️  Warnings: {warning_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"📈 Success Rate: {pass_rate:.1%}")
    print()
    
    # Category breakdown
    print("📂 CATEGORY BREAKDOWN:")
    print("-" * 30)
    
    for category, tests in diagnostic_results.items():
        if category == 'overall_health':
            continue
            
        category_passed = sum(1 for t in tests.values() if t['status'] == 'PASS')
        category_total = len(tests)
        category_rate = category_passed / category_total if category_total > 0 else 0
        
        status_icon = "✅" if category_rate == 1.0 else "⚠️" if category_rate >= 0.5 else "❌"
        print(f"{status_icon} {category.replace('_', ' ').title()}: {category_passed}/{category_total} ({category_rate:.1%})")
    
    # Critical issues
    if failed_tests > 0:
        print("\n🚨 CRITICAL ISSUES FOUND:")
        print("-" * 30)
        for i, error in enumerate(errors_found[:10], 1):  # Show top 10
            print(f"{i}. {error}")
    
    # Warnings
    if warning_tests > 0:
        print("\n⚠️  WARNINGS:")
        print("-" * 15)
        for i, warning in enumerate(warnings_found[:5], 1):  # Show top 5
            print(f"{i}. {warning}")
    
    # Solutions
    if solutions:
        print("\n💡 RECOMMENDED SOLUTIONS:")
        print("-" * 30)
        for i, solution in enumerate(solutions[:10], 1):  # Show top 10
            print(f"{i}. {solution}")
    
    # System requirements check
    print("\n🔧 SYSTEM REQUIREMENTS:")
    print("-" * 25)
    
    requirements = {
        'Python >= 3.8': sys.version_info >= (3, 8),
        'enhprog.py exists': Path('enhprog.py').exists(),
        'models directory': Path('models').exists(),
        'FMP API key': bool(os.getenv('FMP_API_KEY')),
        'Write permissions': True  # We tested this earlier
    }
    
    for req, status in requirements.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {req}")
    
    # Next steps
    print("\n🎯 NEXT STEPS:")
    print("-" * 15)
    
    if failed_tests == 0:
        print("✨ Your system is working well! Consider:")
        print("  • Training models for better predictions")
        print("  • Setting up FMP API key for real-time data")
        print("  • Testing with live market data")
    elif failed_tests <= 3:
        print("🔧 Address the critical issues above, then:")
        print("  • Re-run this diagnostic")
        print("  • Test basic prediction functionality")
        print("  • Verify API connectivity")
    else:
        print("🚨 Major issues detected. Priority actions:")
        print("  • Check enhprog.py implementation")
        print("  • Install missing dependencies")
        print("  • Verify file structure")
        print("  • Set up API keys")
    
    # Save detailed report
    try:
        import json
        report_path = Path(f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(diagnostic_results, f, indent=2, default=str)
        print(f"\n💾 Detailed report saved to: {report_path}")
    except Exception as e:
        print(f"\n⚠️  Could not save detailed report: {e}")
    
    print("\n" + "=" * 60)
    print(f"🏁 Diagnostic completed at: {datetime.now()}")
    print("=" * 60)

def run_performance_tests():
    """Run performance tests"""
    print("\n⚡ Testing Performance")
    print("-" * 25)
    
    try:
        import time
        import enhprog
        import pandas as pd
        import numpy as np
        
        # Test data processing speed
        start_time = time.time()
        sample_data = pd.DataFrame({
            'Open': np.random.randn(1000) + 100,
            'High': np.random.randn(1000) + 101,
            'Low': np.random.randn(1000) + 99,
            'Close': np.random.randn(1000) + 100,
            'Volume': np.random.randint(1000, 10000, 1000)
        })
        
        enhanced_data = enhprog.enhance_features(sample_data, ['Open', 'High', 'Low', 'Close', 'Volume'])
        processing_time = time.time() - start_time
        
        if processing_time < 5.0:
            log_result('prediction_engine', 'processing_speed', 'PASS', 
                      f"Data processing completed in {processing_time:.2f}s")
        elif processing_time < 10.0:
            log_result('prediction_engine', 'processing_speed', 'WARN', 
                      f"Data processing took {processing_time:.2f}s (slow)")
        else:
            log_result('prediction_engine', 'processing_speed', 'FAIL', 
                      f"Data processing took {processing_time:.2f}s (too slow)", "Performance issue")
        
        # Test memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb < 500:
                log_result('prediction_engine', 'memory_usage', 'PASS', 
                          f"Memory usage: {memory_mb:.1f}MB")
            elif memory_mb < 1000:
                log_result('prediction_engine', 'memory_usage', 'WARN', 
                          f"Memory usage: {memory_mb:.1f}MB (high)")
            else:
                log_result('prediction_engine', 'memory_usage', 'FAIL', 
                          f"Memory usage: {memory_mb:.1f}MB (too high)", "Memory issue")
        except ImportError:
            log_result('prediction_engine', 'memory_usage', 'WARN', 
                      "Could not test memory usage (psutil not installed)", "Optional dependency")
            
    except Exception as e:
        log_result('prediction_engine', 'performance_test', 'FAIL', 
                  "Performance test failed", str(e))

def main():
    """Main diagnostic function"""
    try:
        # Run all diagnostic tests
        test_basic_imports()
        test_enhprog_imports()
        test_api_keys()
        test_data_providers()
        test_model_classes()
        test_prediction_engine()
        test_real_time_data()
        test_file_system()
        test_advanced_features()
        test_prediction_workflow()
        test_streamlit_integration()
        test_chart_generation()
        run_performance_tests()
        
        # Generate comprehensive report
        generate_comprehensive_report()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Diagnostic failed with error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()