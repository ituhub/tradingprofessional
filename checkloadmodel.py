import os
import sys
import numpy as np
import torch
import logging

# Ensure the project directory is in the Python path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Import required modules from enhprog
    from enhprog import (
        load_trained_models, 
        get_real_time_prediction,
        MultiTimeframeDataManager,
        enhance_features,
        ENHANCED_TICKERS
    )
except ImportError as e:
    logger.error(f"Import Error: {e}")
    sys.exit(1)

def test_model_loading(ticker='^GSPC'):
    """
    Test loading of trained models for a specific ticker
    
    Args:
        ticker (str): Stock ticker to test. Defaults to S&P 500.
    
    Returns:
        tuple: Loaded models and configuration
    """
    logger.info(f"🔍 Testing Model Loading for {ticker}")
    
    try:
        # Attempt to load models
        models, config = load_trained_models(ticker)
        
        if not models:
            logger.error("❌ No models could be loaded")
            return None, None
        
        logger.info("✅ Models Loaded Successfully:")
        for model_name in models.keys():
            logger.info(f"   - {model_name}")
        
        return models, config
    
    except Exception as e:
        logger.error(f"Model Loading Error: {e}")
        return None, None

def test_model_prediction(ticker='^GSPC'):
    """
    Test generating a real-time prediction
    
    Args:
        ticker (str): Stock ticker to test prediction
    
    Returns:
        dict: Prediction results
    """
    logger.info(f"🚀 Testing Prediction for {ticker}")
    
    try:
        # Generate real-time prediction
        prediction = get_real_time_prediction(ticker)
        
        if not prediction:
            logger.error("❌ Prediction Generation Failed")
            return None
        
        # Log detailed prediction information
        logger.info("✅ Prediction Generated Successfully:")
        logger.info(f"   Current Price: ${prediction.get('current_price', 0):.2f}")
        logger.info(f"   Predicted Price: ${prediction.get('predicted_price', 0):.2f}")
        logger.info(f"   Price Change: {prediction.get('price_change_pct', 0):.2f}%")
        logger.info(f"   Confidence: {prediction.get('confidence', 0):.2f}%")
        logger.info(f"   Models Used: {prediction.get('models_used', [])}")
        
        return prediction
    
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return None

def analyze_model_directory():
    """
    Analyze the models directory and provide insights
    """
    import pathlib

    model_dir = pathlib.Path("models")
    
    if not model_dir.exists():
        logger.error("❌ Models directory not found")
        return
    
    logger.info("🗂️ Model Directory Analysis")
    
    # Count model files
    model_files = list(model_dir.glob('*'))
    file_types = {}
    
    for file in model_files:
        ext = file.suffix
        file_types[ext] = file_types.get(ext, 0) + 1
    
    logger.info("📊 Model File Statistics:")
    for ext, count in file_types.items():
        logger.info(f"   - {ext} files: {count}")
    
    # Identify unique tickers
    tickers = set()
    for file in model_files:
        parts = file.stem.split('_')
        if parts[0].startswith('^'):
            tickers.add(parts[0])
    
    logger.info("🏷️ Trained Tickers:")
    for ticker in sorted(tickers):
        logger.info(f"   - {ticker}")

def main():
    """
    Main diagnostic function to test model loading and prediction
    """
    logger.info("🤖 AI Trading Model Diagnostic Tool")
    logger.info("=" * 40)
    
    # Analyze model directory
    analyze_model_directory()
    
    # Test model loading and prediction for S&P 500
    models, config = test_model_loading('^GSPC')
    
    if models:
        test_model_prediction('^GSPC')
    
    logger.info("\n🏁 Diagnostic Complete")

if __name__ == "__main__":
    main()