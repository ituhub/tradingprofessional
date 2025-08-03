# =============================================================================
# ENHANCED FOOTER SYSTEM WITH DOCUMENTATION, API, AND SUPPORT
# Updated version for AI Trading Professional with full feature coverage
# =============================================================================

import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import json

# =============================================================================
# DOCUMENTATION SYSTEM
# =============================================================================

class DocumentationSystem:
    """Complete documentation system for AI Trading Professional"""
    
    @staticmethod
    def create_documentation_page():
        """Create comprehensive documentation page"""
        st.title("📚 AI Trading Professional - Documentation")
        
        # Navigation tabs
        doc_tabs = st.tabs([
            "🚀 Getting Started", 
            "🎯 Features", 
            "🤖 Models", 
            "📊 Analytics", 
            "⚙️ Settings",
            "❓ FAQ"
        ])
        
        with doc_tabs[0]:
            DocumentationSystem.show_getting_started()
        
        with doc_tabs[1]:
            DocumentationSystem.show_features()
        
        with doc_tabs[2]:
            DocumentationSystem.show_models()
        
        with doc_tabs[3]:
            DocumentationSystem.show_analytics()
        
        with doc_tabs[4]:
            DocumentationSystem.show_settings()
        
        with doc_tabs[5]:
            DocumentationSystem.show_faq()
    
    @staticmethod
    def show_getting_started():
        """Getting started guide"""
        st.header("🚀 Getting Started")
        
        st.markdown("""
        ## Welcome to AI Trading Professional
        
        AI Trading Professional is a comprehensive trading platform that uses advanced machine learning 
        models to provide price predictions and market analysis across multiple asset classes.
        
        ### Quick Start Guide
        
        #### 1. **Choose Your Subscription**
        - **Free Tier**: 10 predictions per day, 2 basic models (XGBoost, Sklearn Ensemble)
        - **Premium Tier**: Unlimited predictions, 8 advanced AI models, real-time data
        
        #### 2. **Select an Asset**
        Navigate to the sidebar and choose from:
        - 📊 **Indices**: S&P 500 (^GSPC), DAX (^GDAXI), HSI (^HSI), SPX (^SPX)
        - 🛢️ **Commodities**: Oil (CC=F), Gold (GC=F), Natural Gas (NG=F), Coffee (KC=F), Silver (SI=F), Copper (HG=F)
        - ₿ **Crypto**: Bitcoin (BTCUSD), Ethereum (ETHUSD), Solana (SOLUSD), Binance (BNBUSD)
        - 💱 **Forex**: USD/JPY (USDJPY) and others
        
        #### 3. **Configure Settings**
        - Select timeframe (15min to 1day - Premium only for sub-daily)
        - Choose AI models (Premium: all 8 models available)
        - Set risk preferences
        
        #### 4. **Get Predictions**
        Click "Get AI Prediction" to receive:
        - Price forecasts with 1-2% realistic changes
        - Trading recommendations with entry/exit levels
        - Risk analysis and confidence scores
        - 5-day price forecasts
        
        ### Premium Key Activation
        
        To unlock all features, enter your premium key in the sidebar:
        ```
        Premium Key: Prem246_357
        ```
        
        ### System Requirements
        - Modern web browser (Chrome, Firefox, Safari, Edge)
        - Stable internet connection for real-time data
        - JavaScript enabled
        - Minimum screen resolution: 1024x768
        
        ### Supported Assets & Realistic Changes
        
        The system is calibrated for realistic price movements:
        - **Crypto**: Up to ±8% daily changes
        - **Forex**: Up to ±2% daily changes  
        - **Commodities**: Up to ±5% daily changes
        - **Indices**: Up to ±3% daily changes
        - **Stocks**: Up to ±6% daily changes
        
        All predictions are constrained within these realistic bounds.
        """)
    
    @staticmethod
    def show_features():
        """Features documentation"""
        st.header("🎯 Features Overview")
        
        # Feature comparison table
        feature_data = {
            "Feature": [
                "Daily Predictions",
                "AI Models Available", 
                "Timeframes",
                "Technical Analysis",
                "Risk Analytics",
                "Backtesting",
                "Portfolio Optimization",
                "Market Regime Detection",
                "Model Drift Detection",
                "Real-time Data",
                "Alternative Data",
                "Export Data",
                "API Access",
                "Cross-Validation",
                "Model Explanations",
                "Sentiment Analysis",
                "Economic Indicators",
                "Options Flow Data",
                "Multi-Asset Support",
                "Trading Plan Generation"
            ],
            "Free Tier": [
                "10 per day",
                "2 basic models",
                "Daily only",
                "Basic indicators",
                "❌",
                "❌", 
                "❌",
                "❌",
                "❌",
                "❌",
                "❌",
                "❌",
                "❌",
                "❌",
                "❌",
                "❌",
                "❌",
                "❌",
                "Limited (4 assets)",
                "Basic levels"
            ],
            "Premium Tier": [
                "Unlimited",
                "8 advanced models",
                "All (15min-1day)",
                "Advanced suite (50+ indicators)",
                "✅ Full analytics",
                "✅ Complete backtesting",
                "✅ Mean-variance optimization",
                "✅ 4 regime types",
                "✅ Real-time monitoring",
                "✅ Live market data",
                "✅ Sentiment & economic",
                "✅ CSV/JSON export",
                "✅ REST API",
                "✅ 5-fold validation",
                "✅ SHAP, permutation",
                "✅ Reddit, Twitter, News",
                "✅ FRED integration",
                "✅ Real-time flow",
                "✅ All asset classes",
                "✅ Advanced risk management"
            ]
        }
        
        st.table(feature_data)
        
        st.markdown("""
        ## Core Features Explained
        
        ### 🤖 AI Prediction Engine
        Our advanced AI system combines multiple machine learning models to provide accurate price predictions:
        
        - **8 Neural Network Models**: Advanced Transformer, CNN-LSTM, Enhanced TCN, Enhanced Informer, Enhanced N-BEATS, LSTM-GRU Ensemble
        - **2 Tree-based Models**: XGBoost, Scikit-learn Ensemble  
        - **Ensemble Learning**: Combines predictions from multiple models with intelligent weighting
        - **Confidence Scoring**: Each prediction includes confidence metrics (45-88% range)
        - **Validation**: All predictions are validated against realistic market constraints
        - **Cross-Validation**: 5-fold time series validation for model selection
        
        ### 📊 Technical Analysis
        Comprehensive technical indicators (50+ for Premium):
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands with position tracking
        - Multiple Moving Averages (SMA, EMA)
        - Volume indicators (OBV, Volume Price Trend)
        - Advanced indicators: Ichimoku Cloud, Stochastic, Williams %R, CCI, ATR
        - Market microstructure features
        
        ### ⚠️ Risk Management
        Professional risk analytics (Premium):
        - Value at Risk (VaR) at 95% and 99% confidence levels
        - Expected Shortfall (Conditional VaR)
        - Sharpe & Sortino Ratios
        - Maximum Drawdown calculation
        - Volatility analysis (realized, GARCH)
        - Calmar Ratio
        - Dynamic position sizing with Kelly Criterion
        
        ### 🌐 Alternative Data (Premium)
        Access to real-time alternative data sources:
        - **Social Media Sentiment**: Reddit, Twitter analysis
        - **News Sentiment**: Real-time news analysis
        - **Economic Indicators**: FRED API integration (10 key indicators)
        - **Options Flow**: Put/call ratios, implied volatility, gamma exposure
        - **Market Microstructure**: Bid-ask spreads, market impact
        
        ### 🎯 Trading Plan Generation
        Intelligent trading plans with:
        - **Entry Levels**: Optimal entry prices based on predictions
        - **Stop Loss**: Dynamic stop levels (typically 1% from entry)
        - **Take Profit**: Multiple targets (1% and 2% levels)
        - **Risk/Reward Ratios**: Calculated for each target
        - **Position Sizing**: Based on account size and risk tolerance
        
        ### 📈 Multi-Timeframe Analysis (Premium)
        - **15 Minutes**: Ultra short-term scalping
        - **1 Hour**: Intraday trading
        - **4 Hours**: Swing trading
        - **1 Day**: Position trading
        
        ### 🔍 Model Explainability (Premium)
        Understanding AI decisions through:
        - **SHAP Values**: Shapley Additive Explanations
        - **Feature Importance**: Tree-based importance scores
        - **Permutation Importance**: Model-agnostic explanations
        - **Gradient-based Importance**: Neural network explanations
        """)
    
    @staticmethod
    def show_models():
        """AI Models documentation"""
        st.header("🤖 AI Models & Algorithms")
        
        st.markdown("""
        ## Available Models
        
        Our platform uses 8 state-of-the-art machine learning models, each optimized for different market conditions and trading scenarios:
        """)
        
        # Model descriptions with updated information
        models_info = {
            "🧠 Advanced Transformer": {
                "Type": "Neural Network",
                "Architecture": "Multi-head attention with positional encoding",
                "Strengths": "Long-term trends, complex pattern recognition, sequence modeling",
                "Best For": "Multi-day forecasts, trend analysis, pattern detection",
                "Confidence": "85%",
                "Bias": "Slightly bullish (+0.5%)",
                "Training": "Supervised learning with time series cross-validation",
                "Parameters": "256 hidden units, 8 attention heads, 6 layers"
            },
            "🔄 CNN-LSTM with Attention": {
                "Type": "Neural Network", 
                "Architecture": "Convolutional layers + LSTM + Multi-head attention",
                "Strengths": "Temporal dependencies, short-term predictions, local patterns",
                "Best For": "Intraday trading, pattern recognition, volatility prediction",
                "Confidence": "75%",
                "Bias": "Slightly bullish (+0.3%)",
                "Training": "End-to-end training with dropout regularization",
                "Parameters": "64-128 conv filters, 100 LSTM units, 4 attention heads"
            },
            "📊 Enhanced TCN": {
                "Type": "Neural Network",
                "Architecture": "Temporal Convolutional Network with dilated convolutions",
                "Strengths": "Local patterns, noise resistance, parallel processing", 
                "Best For": "Volatile markets, feature extraction, real-time inference",
                "Confidence": "70%",
                "Bias": "Slightly bearish (-0.2%)",
                "Training": "Causal convolutions with residual connections",
                "Parameters": "4 levels, kernel size 3, 64-256 channels"
            },
            "🎯 Enhanced Informer": {
                "Type": "Neural Network",
                "Architecture": "Sparse attention transformer for long sequences",
                "Strengths": "Long sequences, attention mechanisms, efficiency",
                "Best For": "Complex dependencies, long-term forecasting",
                "Confidence": "65%", 
                "Bias": "Neutral (0%)",
                "Training": "ProbSparse self-attention with distilling operation",
                "Parameters": "128 hidden units, 8 heads, 3 encoder layers"
            },
            "📈 Enhanced N-BEATS": {
                "Type": "Neural Network",
                "Architecture": "Neural basis expansion analysis for interpretable forecasting",
                "Strengths": "Hierarchical forecasting, interpretable, trend/seasonality decomposition",
                "Best For": "Time series decomposition, interpretable predictions",
                "Confidence": "80%",
                "Bias": "Moderately bullish (+0.7%)",
                "Training": "Backcast and forecast branches with basis functions",
                "Parameters": "6 blocks, 256 hidden units per block"
            },
            "🔀 LSTM-GRU Ensemble": {
                "Type": "Neural Network",
                "Architecture": "Ensemble of LSTM and GRU networks with fusion layer",
                "Strengths": "Ensemble learning, robust predictions, memory mechanisms", 
                "Best For": "Balanced predictions, risk management, stable forecasts",
                "Confidence": "75%",
                "Bias": "Slightly bullish (+0.2%)",
                "Training": "Dual-branch training with weighted fusion",
                "Parameters": "128 units each branch, 2 layers, dropout 0.2"
            },
            "🌳 XGBoost": {
                "Type": "Tree-based",
                "Architecture": "Gradient boosting with extreme gradient boosting",
                "Strengths": "Feature importance, non-linear relationships, robust to outliers",
                "Best For": "Feature analysis, structured data, interpretability",
                "Confidence": "70%",
                "Bias": "Slightly bearish (-0.3%)",
                "Training": "Gradient boosting with regularization",
                "Parameters": "300 estimators, max depth 10, learning rate 0.08"
            },
            "🎲 Scikit-learn Ensemble": {
                "Type": "Tree-based", 
                "Architecture": "Ensemble of Random Forest, Gradient Boosting, SVR, Ridge, Lasso",
                "Strengths": "Robust to overfitting, diverse models, stable predictions",
                "Best For": "Stable predictions, risk aversion, baseline comparisons",
                "Confidence": "65%",
                "Bias": "Neutral (0%)",
                "Training": "Individual model training with ensemble averaging",
                "Parameters": "100 trees RF, 100 estimators GB, RBF kernel SVR"
            }
        }
        
        for model_name, info in models_info.items():
            with st.expander(f"{model_name}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Type:** {info['Type']}")
                    st.markdown(f"**Confidence:** {info['Confidence']}")
                    st.markdown(f"**Bias:** {info['Bias']}")
                    st.markdown(f"**Architecture:** {info['Architecture']}")
                
                with col2:
                    st.markdown(f"**Strengths:** {info['Strengths']}")
                    st.markdown(f"**Best For:** {info['Best For']}")
                    st.markdown(f"**Training:** {info['Training']}")
                    st.markdown(f"**Parameters:** {info['Parameters']}")
        
        st.markdown("""
        ## Ensemble Methodology
        
        ### Voting System
        Our ensemble approach combines multiple models using advanced techniques:
        
        1. **Weighted Averaging**: Models weighted by historical performance and confidence
        2. **Cross-Validation Weights**: Weights determined by 5-fold time series CV performance
        3. **Bias Correction**: Systematic biases are identified and corrected
        4. **Outlier Detection**: Extreme predictions are filtered using statistical bounds
        5. **Consensus Building**: Final prediction uses weighted consensus with validation
        
        ### Model Selection Guidelines
        
        **For Crypto Trading (High Volatility):**
        - CNN-LSTM for short-term (handles volatility)
        - Enhanced TCN for noise resistance
        - XGBoost for feature analysis
        - Expected daily changes: ±2-8%
        
        **For Forex Trading (Low Volatility):**
        - Enhanced Informer for macro trends
        - Scikit-learn Ensemble for stability
        - Advanced Transformer for pattern recognition
        - Expected daily changes: ±0.5-2%
        
        **For Stocks/Indices (Moderate Volatility):**
        - Advanced Transformer for trend analysis
        - N-BEATS for interpretable forecasts
        - LSTM-GRU Ensemble for balanced approach
        - Expected daily changes: ±1-6%
        
        **For Commodities (Variable Volatility):**
        - Enhanced TCN for noise resistance
        - XGBoost for supply/demand factors
        - LSTM-GRU Ensemble for robust predictions
        - Expected daily changes: ±1-5%
        
        ### Performance Metrics
        
        Our models are evaluated using multiple metrics:
        - **Mean Squared Error (MSE)**: Primary optimization target
        - **Mean Absolute Error (MAE)**: Robustness measure
        - **Directional Accuracy**: Percentage of correct price direction predictions
        - **Sharpe Ratio**: Risk-adjusted returns in backtesting
        - **Maximum Drawdown**: Worst-case loss scenarios
        
        ### Model Training Process
        
        1. **Data Collection**: Real-time and historical data from multiple sources
        2. **Feature Engineering**: 50+ technical and fundamental features
        3. **Cross-Validation**: 5-fold time series split for robust evaluation
        4. **Hyperparameter Tuning**: Grid search with validation-based selection
        5. **Ensemble Construction**: Weight optimization based on CV performance
        6. **Drift Detection**: Continuous monitoring for model degradation
        7. **Retraining**: Automated retraining when drift is detected
        """)
    
    @staticmethod
    def show_analytics():
        """Analytics documentation"""
        st.header("📊 Analytics & Interpretation")
        
        st.markdown("""
        ## Understanding Your Results
        
        ### Prediction Confidence Levels
        
        Our AI system provides confidence scores between 45-88%:
        
        - **🟢 High Confidence (80-88%)**: Strong model agreement, clear market signals
        - **🟡 Moderate Confidence (60-79%)**: Good signal quality, some uncertainty
        - **🔴 Low Confidence (45-59%)**: Conflicting signals, high uncertainty
        
        **Note**: We cap confidence at 88% to maintain realistic expectations. No AI system should claim >90% certainty in financial markets.
        
        ### Price Change Interpretation
        
        **Realistic Daily Change Ranges by Asset:**
        - **Crypto**: Typical ±2-8% daily moves (max constraint: ±8%)
        - **Forex**: Typical ±0.5-2% daily moves (max constraint: ±2%)
        - **Stocks**: Typical ±1-6% daily moves (max constraint: ±6%)
        - **Commodities**: Typical ±1-5% daily moves (max constraint: ±5%)
        - **Indices**: Typical ±0.8-3% daily moves (max constraint: ±3%)
        
        All predictions are automatically constrained within these realistic bounds.
        
        ### Risk Metrics Explained (Premium)
        
        #### Value at Risk (VaR)
        - **Definition**: Maximum expected loss over a given time period at specified confidence
        - **95% VaR**: Loss threshold exceeded only 5% of the time
        - **99% VaR**: Loss threshold exceeded only 1% of the time
        - **Interpretation**: Lower absolute values indicate lower risk
        
        #### Expected Shortfall (Conditional VaR)
        - **Definition**: Average loss when VaR threshold is exceeded
        - **Usage**: More comprehensive risk measure than VaR alone
        - **Interpretation**: Shows severity of tail risk
        
        #### Sharpe Ratio
        - **Excellent**: > 2.0 (rare in practice)
        - **Good**: 1.0 - 2.0  
        - **Acceptable**: 0.5 - 1.0
        - **Poor**: < 0.5
        - **Formula**: (Return - Risk-free rate) / Volatility
        
        #### Sortino Ratio
        - **Improvement over Sharpe**: Only penalizes downside volatility
        - **Good**: > 1.5
        - **Acceptable**: 0.8 - 1.5
        - **Poor**: < 0.8
        
        #### Maximum Drawdown
        - **Low Risk**: < 10%
        - **Moderate Risk**: 10-20%
        - **High Risk**: 20-30%
        - **Very High Risk**: > 30%
        - **Definition**: Largest peak-to-trough decline
        
        ### Trading Plan Components
        
        #### Entry Signals
        - **Buy Signal**: AI predicts price increase with realistic targets
        - **Sell Signal**: AI predicts price decrease with realistic targets
        - **Hold Signal**: Unclear direction or low confidence
        
        #### Risk Management
        - **Stop Loss**: Typically 1% from entry to limit losses
        - **Take Profit Level 1**: Typically 1% target for partial profits
        - **Take Profit Level 2**: Typically 2% target for remaining position
        - **Risk/Reward Ratios**: Calculated for each target level
        - **Position Sizing**: Based on account size and risk tolerance
        
        ### Market Regime Analysis (Premium)
        
        #### Regime Types
        1. **Bull Market**: Sustained upward trend, positive sentiment, low volatility
        2. **Bear Market**: Sustained downward trend, negative sentiment, high volatility
        3. **Sideways/Consolidation**: Range-bound, indecisive direction, moderate volatility
        4. **High Volatility**: Large price swings, uncertainty, extreme movements
        
        #### Strategy Adjustments by Regime
        - **Bull Market**: Trend-following strategies, longer holds, higher position sizes
        - **Bear Market**: Defensive strategies, shorter timeframes, reduced positions
        - **Sideways**: Range trading, mean reversion, neutral positioning
        - **High Volatility**: Reduced position sizes, tighter stops, increased monitoring
        
        ### Alternative Data Integration (Premium)
        
        #### Sentiment Analysis
        - **Reddit Sentiment**: Extracted from investing and asset-specific subreddits
        - **Twitter Sentiment**: Real-time social media sentiment analysis
        - **News Sentiment**: Professional news source sentiment scoring
        - **Range**: -1 (very bearish) to +1 (very bullish)
        
        #### Economic Indicators
        - **Interest Rates**: 10-Year Treasury, Fed Funds Rate
        - **Employment**: Unemployment Rate
        - **Inflation**: CPI data
        - **Growth**: GDP indicators
        - **Currency**: Exchange rates
        - **Commodities**: Oil and gold prices
        
        #### Options Flow (Premium)
        - **Put/Call Ratio**: Sentiment indicator from options activity
        - **Implied Volatility**: Market expectation of future volatility
        - **Gamma Exposure**: Market maker positioning effects
        - **Dark Pool Index**: Institutional trading activity
        - **Max Pain**: Options expiration pressure points
        
        ### Model Explanations (Premium)
        
        #### SHAP Values
        - **Purpose**: Shows feature contribution to individual predictions
        - **Interpretation**: Positive values increase prediction, negative decrease
        - **Usage**: Understand which factors drive each prediction
        
        #### Feature Importance
        - **Tree Models**: Built-in importance from XGBoost and Random Forest
        - **Neural Networks**: Gradient-based and permutation importance
        - **Top Features**: Usually price-based and volume indicators
        
        #### Permutation Importance
        - **Method**: Measures prediction change when feature is randomized
        - **Advantage**: Model-agnostic, works with any model type
        - **Interpretation**: Higher values indicate more important features
        """)
    
    @staticmethod
    def show_settings():
        """Settings and configuration"""
        st.header("⚙️ Settings & Configuration")
        
        st.markdown("""
        ## Platform Configuration
        
        ### Subscription Tiers
        
        #### Free Tier Features
        - 10 predictions per day
        - 2 AI models (XGBoost, Scikit-learn Ensemble)
        - Daily timeframe only
        - Basic technical indicators
        - Limited to 4 asset categories
        - Basic trading plan generation
        
        #### Premium Tier Features (Key: Prem246_357)
        - Unlimited predictions
        - 8 advanced AI models
        - All timeframes (15min to 1day)
        - 50+ technical indicators
        - Complete risk analytics suite
        - Model explanations and drift detection
        - Real-time alternative data
        - Portfolio optimization
        - Market regime analysis
        - Backtesting capabilities
        - API access
        
        ### Asset Selection
        
        Choose from 4 major asset categories with 15 total assets:
        
        **📊 Indices (4 assets)**
        - ^GSPC: S&P 500 Index
        - ^GDAXI: DAX (German Stock Index)
        - ^HSI: Hang Seng Index
        - ^SPX: S&P 500 Futures
        
        **🛢️ Commodities (6 assets)** 
        - GC=F: Gold Futures
        - CC=F: Crude Oil Futures
        - NG=F: Natural Gas Futures
        - KC=F: Coffee Futures
        - SI=F: Silver Futures
        - HG=F: Copper Futures
        
        **₿ Cryptocurrencies (4 assets)**
        - BTCUSD: Bitcoin
        - ETHUSD: Ethereum  
        - SOLUSD: Solana
        - BNBUSD: Binance Coin
        
        **💱 Forex (1 asset)**
        - USDJPY: US Dollar/Japanese Yen
        
        ### Timeframe Selection
        
        **Available Timeframes (Premium only for intraday):**
        - **15min**: Ultra short-term scalping (Premium)
        - **1hour**: Short-term intraday (Premium)
        - **4hour**: Swing trading (Premium)
        - **1day**: Position trading (All tiers)
        
        **Timeframe Guidelines:**
        - **Day Trading**: Use 15min-1hour timeframes
        - **Swing Trading**: Use 4hour-1day timeframes
        - **Position Trading**: Use 1day timeframe
        
        ### Model Configuration (Premium)
        
        **Recommended Model Combinations:**
        
        *Conservative Portfolio (Stability Focus):*
        - Scikit-learn Ensemble (stable baseline)
        - Enhanced Informer (trend analysis)
        - XGBoost (feature insights)
        
        *Aggressive Portfolio (Performance Focus):*
        - Advanced Transformer (pattern recognition)
        - CNN-LSTM (short-term precision)
        - N-BEATS (trend decomposition)
        
        *Balanced Portfolio (Risk-Adjusted):*
        - LSTM-GRU Ensemble (robust predictions)
        - Enhanced TCN (noise resistance)
        - XGBoost (non-linear relationships)
        - Scikit-learn Ensemble (diversification)
        
        ### Risk Settings
        
        **Risk Tolerance Levels:**
        - **Conservative**: 1% stop loss, 1-2% targets, smaller positions
        - **Moderate**: 1% stop loss, 1-3% targets, standard positions
        - **Aggressive**: 1.5% stop loss, 2-4% targets, larger positions
        
        **Position Sizing Guidelines:**
        - **Free Tier**: Suggested 2-5% of account per trade
        - **Premium**: Kelly Criterion-based dynamic sizing
        - **Maximum**: Never risk more than 10% on single trade
        
        ### Data Sources & Quality
        
        **Real-time Data Providers:**
        - Financial Modeling Prep (FMP) - Primary price data
        - Federal Reserve Economic Data (FRED) - Economic indicators
        - Social media APIs - Sentiment data
        - News aggregators - Sentiment analysis
        
        **Data Quality Monitoring:**
        - Real-time validation of price data
        - Cross-reference multiple sources
        - Automatic fallback to demo mode if data issues
        - Quality scoring based on freshness, completeness, accuracy
        
        ### Performance Optimization
        
        **System Settings:**
        - **Cache Duration**: 30 seconds (market hours), 5 minutes (after hours)
        - **Model Loading**: Lazy loading for optimal performance
        - **Prediction Caching**: Results cached for 60 seconds
        - **Auto-refresh**: Real-time price updates every 10 seconds (Premium)
        
        **Browser Requirements:**
        - Modern browser with JavaScript enabled
        - Minimum 1024x768 screen resolution
        - Stable internet connection for real-time features
        - 4GB RAM recommended for optimal performance
        
        ## Troubleshooting
        
        ### Common Issues & Solutions
        
        **"Daily prediction limit reached"**
        - Solution: Upgrade to Premium or wait for daily reset
        - Workaround: System automatically switches to demo mode
        
        **"Backend temporarily unavailable"**
        - Solution: System uses enhanced demo mode
        - Features: All functionality available in simulation
        
        **Large prediction changes (>5%)**
        - Automatic: System constrains to realistic bounds
        - Check: Asset type and typical volatility
        - Review: Model selection and confidence scores
        
        **Low confidence scores (<60%)**
        - Check: Market conditions and volatility
        - Try: Different model combinations
        - Consider: Longer timeframes for better signals
        
        **Model training failures**
        - Requirement: Premium subscription needed
        - Check: Sufficient historical data available
        - Retry: With different model selections
        
        ### Performance Optimization Tips
        
        1. **For Best Accuracy**: Use all 8 models in Premium
        2. **For Speed**: Select 2-3 fastest models (XGBoost, Sklearn)
        3. **For Stability**: Include ensemble models (LSTM-GRU, Sklearn)
        4. **For Insights**: Include explainable models (XGBoost, N-BEATS)
        
        ### Data Privacy & Security
        
        - **No Personal Data**: System doesn't store trading positions
        - **Anonymous Usage**: No account registration required
        - **Local Processing**: Most calculations done client-side
        - **Secure Connections**: All data transmitted over HTTPS
        - **API Keys**: Optional real-time data enhancement
        """)
    
    @staticmethod
    def show_faq():
        """Frequently Asked Questions"""
        st.header("❓ Frequently Asked Questions")
        
        faqs = [
            {
                "question": "How accurate are the AI predictions?",
                "answer": """
                Our AI models achieve varying accuracy depending on market conditions and asset types:
                
                **Directional Accuracy:**
                - **Typical Range**: 65-85% directional accuracy
                - **High Confidence Predictions**: Often 75-85% accurate
                - **Volatile Markets**: Lower accuracy (60-70%)
                - **Stable Markets**: Higher accuracy (75-85%)
                
                **Price Precision:**
                - Predictions constrained to realistic daily changes
                - Cross-validated using 5-fold time series splits
                - Ensemble approach reduces individual model errors
                
                **Important**: No prediction system is 100% accurate. Always use proper risk management and never risk more than you can afford to lose.
                """
            },
            {
                "question": "What's the difference between Free and Premium tiers?",
                "answer": """
                **Free Tier Limitations:**
                - 10 predictions per day (resets at midnight)
                - 2 basic AI models (XGBoost, Scikit-learn Ensemble)
                - Daily timeframe only
                - Basic technical analysis (RSI, MACD, Bollinger Bands)
                - Limited to 4 asset categories
                - Basic trading plan generation
                - No real-time data feeds
                - No model explanations or risk analytics
                
                **Premium Tier Benefits (Key: Prem246_357):**
                - Unlimited predictions
                - 8 advanced AI models including neural networks
                - All timeframes (15min, 1hour, 4hour, 1day)
                - 50+ advanced technical indicators
                - Complete risk analytics suite
                - Model explanations (SHAP, feature importance)
                - Real-time market data and alternative data
                - Portfolio optimization and backtesting
                - Market regime analysis and drift detection
                - API access for automated trading
                - Export capabilities (CSV, JSON)
                
                **Value Proposition**: Premium offers 10x more features for serious traders.
                """
            },
            {
                "question": "How do I interpret confidence scores?",
                "answer": """
                Confidence scores indicate how certain our AI ensemble is about its prediction:
                
                **Confidence Scale (45-88% range):**
                - **85-88%**: Extremely high confidence - very strong model agreement
                - **80-84%**: High confidence - strong signals across models
                - **70-79%**: Good confidence - reliable signal with some uncertainty
                - **60-69%**: Moderate confidence - proceed with caution
                - **50-59%**: Low confidence - high uncertainty, consider avoiding
                - **45-49%**: Very low confidence - conflicting model signals
                
                **Factors Affecting Confidence:**
                - Model agreement (higher agreement = higher confidence)
                - Market volatility (high volatility = lower confidence)
                - Data quality (better data = higher confidence)
                - Asset type (forex typically higher than crypto)
                - Timeframe (longer timeframes often more confident)
                
                **Trading Guidelines:**
                - Only trade predictions with >60% confidence
                - Use smaller position sizes for 60-70% confidence
                - Use standard sizes for 70-80% confidence
                - Consider larger sizes for >80% confidence (with proper risk management)
                
                **Important**: Higher confidence doesn't guarantee accuracy, but indicates stronger model agreement and historically better performance.
                """
            },
            {
                "question": "Which models should I use for different assets?",
                "answer": """
                **For Cryptocurrencies (High Volatility):**
                - **Primary**: CNN-LSTM (handles high volatility and noise)
                - **Secondary**: Enhanced TCN (robust to extreme movements)
                - **Analysis**: XGBoost (feature importance for market drivers)
                - **Expected Changes**: ±2-8% daily (constrained to realistic bounds)
                
                **For Forex (Low Volatility, Trend-Following):**
                - **Primary**: Enhanced Informer (captures macro trends)
                - **Secondary**: Advanced Transformer (pattern recognition)
                - **Stability**: Scikit-learn Ensemble (consistent performance)
                - **Expected Changes**: ±0.5-2% daily
                
                **For Stocks/Indices (Moderate Volatility):**
                - **Primary**: Advanced Transformer (trend analysis and patterns)
                - **Secondary**: N-BEATS (interpretable trend decomposition)
                - **Balance**: LSTM-GRU Ensemble (robust predictions)
                - **Expected Changes**: ±1-6% daily
                
                **For Commodities (Variable Volatility):**
                - **Primary**: Enhanced TCN (noise resistance for supply/demand shocks)
                - **Secondary**: XGBoost (captures fundamental factors)
                - **Stability**: LSTM-GRU Ensemble (handles volatility regimes)
                - **Expected Changes**: ±1-5% daily
                
                **Universal Recommendations:**
                - Always include at least one ensemble model for stability
                - Use XGBoost for feature importance insights
                - Combine neural networks with tree-based models
                - For best results: use all 8 models (Premium)
                """
            },
            {
                "question": "How often should I get new predictions?",
                "answer": """
                Prediction frequency depends on your trading style and timeframe:
                
                **Day Trading (15min-1hour timeframes):**
                - **Frequency**: Every 30-60 minutes during active trading
                - **Triggers**: Before major news events, significant price moves
                - **Usage**: Quick entries/exits with tight stops
                - **Risk**: Higher frequency = higher transaction costs
                
                **Swing Trading (4hour-1day timeframes):**
                - **Frequency**: 1-2 times per day
                - **Timing**: Before market open/close, after major events
                - **Usage**: Multi-day position holding
                - **Focus**: Wait for high-confidence signals (>70%)
                
                **Position Trading (1day timeframe):**
                - **Frequency**: Weekly or after significant market events
                - **Usage**: Long-term position building
                - **Focus**: High-confidence signals (>75%) with strong fundamentals
                
                **Best Practices:**
                - Don't over-trade based on every prediction
                - Wait for confidence scores >60% minimum
                - Consider transaction costs in frequent trading
                - Use prediction changes as confirmation, not sole triggers
                - Set specific times for analysis (e.g., daily at market close)
                
                **Premium Advantage**: Real-time updates allow for more responsive trading
                """
            },
            {
                "question": "What do I do if predictions seem inaccurate?",
                "answer": """
                If predictions consistently underperform expectations:
                
                **Immediate Checks:**
                1. **Verify Market Conditions**: High volatility reduces all model accuracy
                2. **Check Confidence Scores**: Low confidence (<60%) predictions are naturally less reliable
                3. **Review Timeframe**: Longer timeframes often provide more reliable signals
                4. **Examine Asset Type**: Some assets are inherently harder to predict
                
                **Model Adjustments:**
                1. **Try Different Combinations**: Experiment with various model ensembles
                2. **Focus on High-Confidence**: Only trade predictions >70% confidence
                3. **Use Ensemble Approach**: Combine multiple models for better consensus
                4. **Check Model Bias**: Some models have slight bullish/bearish tendencies
                
                **Market Analysis:**
                1. **Regime Detection**: Check if market regime has changed
                2. **Alternative Data**: Review sentiment and economic indicators (Premium)
                3. **Volatility Assessment**: High volatility periods are harder to predict
                4. **News Events**: Major events can invalidate technical predictions
                
                **Risk Management:**
                1. **Reduce Position Sizes**: Use smaller positions during uncertain periods
                2. **Tighten Stops**: Use closer stop losses to limit losses
                3. **Diversify Timeframes**: Don't rely on single timeframe
                4. **Paper Trade First**: Test strategies without real money
                
                **System Features:**
                - **Drift Detection** (Premium): Automatically identifies when models need updating
                - **Regime Analysis** (Premium): Shows when market conditions change
                - **Model Explanations** (Premium): Understand what drives predictions
                
                **Remember**: Even the best systems have losing periods. Focus on long-term performance and proper risk management.
                """
            },
            {
                "question": "How do I activate Premium features?",
                "answer": """
                **Activation Steps:**
                1. Navigate to the sidebar in the application
                2. Find the "🔑 Subscription" section
                3. Enter Premium Key: `Prem246_357`
                4. Click "Activate Premium" button
                5. Page will refresh with "PREMIUM" badge displayed
                6. All advanced features immediately available
                
                **Verification:**
                - Header shows "PREMIUM" instead of "FREE"
                - Sidebar shows "✅ Premium Active - All features unlocked"
                - Model selection shows all 8 AI models
                - Timeframe selection includes 15min, 1hour, 4hour options
                - Additional tabs appear in prediction results
                
                **Premium Features Unlocked:**
                - Unlimited daily predictions
                - 8 advanced AI models (vs 2 in free)
                - All timeframes (15min to 1day)
                - Risk analytics and portfolio optimization
                - Market regime detection
                - Model drift detection
                - Real-time alternative data feeds
                - Model explanations (SHAP, feature importance)
                - Backtesting capabilities
                - Advanced charting and visualizations
                
                **Troubleshooting:**
                - If activation fails, check key spelling exactly: `Prem246_357`
                - Refresh browser page after activation
                - Clear browser cache if issues persist
                - Key is case-sensitive and must match exactly
                
                **Note**: Premium key is provided for demonstration and educational purposes.
                """
            },
            {
                "question": "Is my data secure and private?",
                "answer": """
                We prioritize data security and user privacy:
                
                **Data Protection:**
                - **No Personal Trading Data Stored**: We don't store your actual trading positions or account information
                - **Anonymous Usage**: No account registration or personal information required
                - **Local Processing**: Most calculations performed client-side in your browser
                - **Session-Based**: Data cleared when you close the browser
                
                **Connection Security:**
                - **HTTPS Encryption**: All data transmitted over secure connections
                - **API Security**: External data requests use encrypted channels
                - **No Data Logging**: We don't log or store prediction requests
                
                **Privacy Features:**
                - **No Tracking**: No user tracking or analytics collection
                - **No Cookies**: Minimal use of browser storage
                - **Open Source**: Core algorithms can be inspected
                - **Transparent**: No hidden data collection
                
                **Real-Time Data:**
                - **Optional**: Real-time features can be disabled
                - **API Keys**: Used only for enhanced data, not stored permanently
                - **Cached Locally**: Market data cached in browser, not on servers
                
                **Recommendations:**
                - Use demo mode for testing strategies
                - Never enter actual account credentials
                - Clear browser data after sessions if desired
                - Consider using incognito/private browsing mode
                
                **What We Don't Collect:**
                - Personal identifying information
                - Trading account details
                - Actual trading positions
                - Financial account information
                - Browsing history outside the application
                
                **Compliance**: We follow industry best practices for financial application security.
                """
            },
            {
                "question": "How do trading plans and risk levels work?",
                "answer": """
                Our AI generates comprehensive trading plans with every prediction:
                
                **Trading Plan Components:**
                - **Entry Price**: Current market price when prediction is made
                - **Strategy**: BUY (bullish prediction) or SELL (bearish prediction)
                - **Direction**: Long Position (expecting price increase) or Short Position (expecting decrease)
                
                **Risk Management Levels:**
                - **Stop Loss**: Typically 1% from entry price to limit losses
                - **Target 1**: First profit target, usually 1% from entry
                - **Target 2**: Second profit target, usually 2% from entry
                - **Risk/Reward Ratios**: Calculated for each target level
                
                **Example Trading Plan (Bullish):**
                ```
                Entry Price: $100.00
                Stop Loss: $99.00 (1% risk)
                Target 1: $101.00 (1% reward)
                Target 2: $102.00 (2% reward)
                Risk/Reward 1: 1:1 ratio
                Risk/Reward 2: 1:2 ratio
                ```
                
                **Position Sizing Guidelines:**
                - **Conservative**: Risk 1% of account per trade
                - **Moderate**: Risk 2% of account per trade
                - **Aggressive**: Risk 3-5% of account per trade
                - **Maximum**: Never risk more than 10% on single trade
                
                **Asset-Specific Adjustments:**
                The system automatically adjusts levels based on asset volatility:
                - **Crypto**: Wider stops due to higher volatility
                - **Forex**: Tighter levels due to lower volatility
                - **Stocks**: Standard levels for moderate volatility
                - **Commodities**: Variable based on specific commodity
                
                **Premium Enhancements:**
                - Dynamic position sizing using Kelly Criterion
                - Advanced risk metrics (VaR, Sharpe ratio)
                - Portfolio-level risk management
                - Backtested performance metrics
                
                **Implementation Tips:**
                1. Always use stop losses - never trade without them
                2. Take partial profits at Target 1
                3. Trail stop loss when Target 1 is hit
                4. Adjust position size based on confidence score
                5. Never risk more than you can afford to lose
                """
            },
            {
                "question": "What makes this system different from other trading tools?",
                "answer": """
                **Advanced AI Architecture:**
                - **8 Sophisticated Models**: Including advanced transformers, CNN-LSTM, and ensemble methods
                - **Cross-Validation**: 5-fold time series validation ensures robust model selection
                - **Realistic Constraints**: All predictions bounded by actual market volatility
                - **Ensemble Intelligence**: Multiple models vote for consensus predictions
                
                **Comprehensive Risk Management:**
                - **Professional Metrics**: VaR, Sharpe ratio, maximum drawdown calculations
                - **Dynamic Position Sizing**: Kelly Criterion-based optimal position sizing
                - **Multi-Asset Calibration**: Different risk parameters for each asset class
                - **Real-Time Monitoring**: Continuous risk assessment and alerts
                
                **Real-Time Alternative Data (Premium):**
                - **Social Sentiment**: Reddit, Twitter sentiment analysis
                - **Economic Integration**: Federal Reserve economic data (FRED API)
                - **Options Flow**: Put/call ratios, implied volatility, gamma exposure
                - **News Analysis**: Real-time news sentiment processing
                
                **Transparency & Explainability:**
                - **Model Explanations**: SHAP values, feature importance, permutation analysis
                - **Confidence Scoring**: Honest 45-88% confidence range (no false 95%+ claims)
                - **Open Methodology**: Clear documentation of all techniques used
                - **Performance Tracking**: Track and display actual model performance
                
                **Educational Focus:**
                - **Learning Platform**: Designed to teach trading concepts while providing predictions
                - **Risk Education**: Emphasizes proper risk management over pure profit
                - **Model Understanding**: Users learn how AI models work
                - **Demo Mode**: Safe environment to learn without financial risk
                
                **Technical Innovation:**
                - **Cross-Validation Framework**: Scientific approach to model evaluation
                - **Drift Detection**: Automatically identifies when models need updates
                - **Regime Detection**: Identifies changing market conditions
                - **Multi-Timeframe Analysis**: Consistent predictions across different time horizons
                
                **Practical Design:**
                - **Realistic Expectations**: No false promises of guaranteed profits
                - **User-Friendly**: Complex AI made accessible to all skill levels
                - **Free Tier Available**: Core functionality accessible without payment
                - **Professional Grade**: Enterprise-level features for serious traders
                
                **What We Don't Do:**
                - Make unrealistic accuracy claims (>90%)
                - Promise guaranteed profits
                - Use black-box algorithms without explanation
                - Ignore proper risk management principles
                - Charge for basic prediction functionality
                
                **Bottom Line**: We combine cutting-edge AI with sound financial principles to create a tool that both educates and assists traders in making informed decisions.
                """
            }
        ]
        
        for faq in faqs:
            with st.expander(f"❓ {faq['question']}", expanded=False):
                st.markdown(faq['answer'])

# =============================================================================
# API DOCUMENTATION SYSTEM
# =============================================================================

class APIDocumentation:
    """Complete API documentation system"""
    
    @staticmethod
    def create_api_documentation():
        """Create API documentation page"""
        st.title("🔌 API Documentation")
        
        api_tabs = st.tabs([
            "🚀 Quick Start",
            "🔐 Authentication", 
            "📡 Endpoints",
            "📝 Examples",
            "📊 Response Format",
            "⚠️ Rate Limits"
        ])
        
        with api_tabs[0]:
            APIDocumentation.show_api_quickstart()
        
        with api_tabs[1]:
            APIDocumentation.show_authentication()
        
        with api_tabs[2]:
            APIDocumentation.show_endpoints()
        
        with api_tabs[3]:
            APIDocumentation.show_examples()
        
        with api_tabs[4]:
            APIDocumentation.show_response_format()
        
        with api_tabs[5]:
            APIDocumentation.show_rate_limits()
    
    @staticmethod
    def show_api_quickstart():
        """API Quick Start Guide"""
        st.header("🚀 API Quick Start")
        
        st.markdown("""
        ## Getting Started with the AI Trading Professional API
        
        The AI Trading Professional API provides programmatic access to our advanced prediction models 
        and analytics suite for automated trading systems and custom applications.
        
        ### Base URL
        ```
        https://api.aitradingpro.com/v1
        ```
        
        ### Quick Example
        ```python
        import requests
        
        # Get a prediction for Bitcoin
        response = requests.post(
            'https://api.aitradingpro.com/v1/predict',
            headers={
                'Authorization': 'Bearer YOUR_API_KEY',
                'Content-Type': 'application/json'
            },
            json={
                'ticker': 'BTCUSD',
                'timeframe': '1day',
                'models': ['advanced_transformer', 'xgboost'],
                'include_trading_plan': True,
                'include_risk_metrics': True
            }
        )
        
        prediction = response.json()
        print(f"Predicted price: ${prediction['data']['predicted_price']:.2f}")
        print(f"Confidence: {prediction['data']['confidence']:.1f}%")
        print(f"Trading strategy: {prediction['data']['trading_plan']['strategy']}")
        ```
        
        ### Available SDKs
        
        **Python SDK**
        ```bash
        pip install aitradingpro-python
        ```
        
        **JavaScript/TypeScript SDK**
        ```bash
        npm install aitradingpro-js
        ```
        
        **R Package**
        ```r
        install.packages("aitradingpro")
        ```
        
        **Go Module**
        ```bash
        go get github.com/aitradingpro/go-sdk
        ```
        
        ### Authentication Requirements
        
        All API requests require authentication using an API key:
        
        1. **Upgrade to Premium** using key: `Prem246_357`
        2. **Generate API Key** in the dashboard settings
        3. **Include in Headers** as Bearer token for all requests
        4. **Monitor Usage** through the dashboard
        
        ### Rate Limits by Tier
        
        - **Free Tier**: 100 requests/hour (same daily prediction limits apply)
        - **Premium Tier**: 10,000 requests/hour with burst capacity
        - **Enterprise**: Custom limits and dedicated infrastructure available
        
        ### Supported Assets
        
        API supports all 15 assets available in the web interface:
        - **4 Indices**: ^GSPC, ^GDAXI, ^HSI, ^SPX
        - **6 Commodities**: GC=F, CC=F, NG=F, KC=F, SI=F, HG=F  
        - **4 Cryptocurrencies**: BTCUSD, ETHUSD, SOLUSD, BNBUSD
        - **1 Forex**: USDJPY
        
        ### Response Times
        
        - **Simple Predictions**: < 500ms
        - **Full Analytics**: < 2 seconds
        - **Batch Requests**: < 5 seconds (up to 10 assets)
        - **Backtesting**: < 30 seconds (depending on parameters)
        
        ### Error Handling
        
        The API uses standard HTTP status codes and returns detailed error messages:
        
        ```python
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raises exception for 4xx/5xx
            data = response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response: {e.response.json()}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        ```
        """)
    
    @staticmethod
    def show_authentication():
        """Authentication documentation"""
        st.header("🔐 Authentication")
        
        st.markdown("""
        ## API Key Authentication
        
        The AI Trading Professional API uses API keys for secure authentication. All requests must include 
        a valid API key in the Authorization header.
        
        ### Obtaining Your API Key
        
        1. **Upgrade to Premium** 
           - Enter premium key: `Prem246_357` in the web interface
           - Activate premium subscription
        
        2. **Access API Settings**
           - Navigate to Settings → API Management
           - Click "Generate New API Key"
           - Set permissions and usage limits
        
        3. **Secure Your Key**
           - Copy the generated key immediately (shown only once)
           - Store securely in environment variables
           - Never commit to version control
        
        ### Authentication Header Format
        ```
        Authorization: Bearer atp_1234567890abcdef...
        ```
        
        ### Example Authentication
        ```bash
        curl -X POST "https://api.aitradingpro.com/v1/predict" \
             -H "Authorization: Bearer atp_1234567890abcdef..." \
             -H "Content-Type: application/json" \
             -d '{
               "ticker": "BTCUSD",
               "timeframe": "1day"
             }'
        ```
        
        ### API Key Types
        
        **Standard API Key**
        - Full access to all prediction endpoints
        - Read-only access to account information
        - Rate limits based on subscription tier
        
        **Limited API Key** 
        - Restricted to specific endpoints
        - Custom rate limits
        - Suitable for third-party integrations
        
        **Webhook API Key**
        - Specialized for receiving webhooks
        - Write access to notification preferences
        - Enhanced security for automated systems
        
        ### Security Best Practices
        
        ✅ **Recommended:**
        - Store API keys in environment variables
        - Use different keys for different environments (dev/prod)
        - Rotate keys regularly (monthly/quarterly)
        - Monitor API usage for anomalies
        - Use HTTPS for all requests
        - Implement proper error handling
        - Set appropriate request timeouts
        
        ❌ **Avoid:**
        - Hardcoding keys in source code
        - Sharing keys in public repositories
        - Using keys in client-side applications
        - Logging keys in application logs
        - Using the same key across multiple applications
        
        ### Key Management
        
        **Rotation Policy:**
        ```python
        import os
        import requests
        from datetime import datetime
        
        class APIKeyManager:
            def __init__(self):
                self.primary_key = os.getenv('ATP_PRIMARY_KEY')
                self.backup_key = os.getenv('ATP_BACKUP_KEY')
                self.last_rotation = os.getenv('ATP_LAST_ROTATION')
            
            def get_active_key(self):
                # Implement key rotation logic
                if self.should_rotate():
                    return self.backup_key
                return self.primary_key
            
            def should_rotate(self):
                # Check if 30 days since last rotation
                if not self.last_rotation:
                    return False
                last_date = datetime.fromisoformat(self.last_rotation)
                return (datetime.now() - last_date).days > 30
        ```
        
        ### Error Responses
        
        **401 Unauthorized - Invalid Key**
        ```json
        {
          "success": false,
          "error": {
            "code": "INVALID_API_KEY",
            "message": "The provided API key is invalid or expired",
            "details": {
              "key_status": "invalid",
              "suggestion": "Generate a new API key in the dashboard"
            }
          }
        }
        ```
        
        **403 Forbidden - Insufficient Permissions**  
        ```json
        {
          "success": false,
          "error": {
            "code": "INSUFFICIENT_PERMISSIONS",
            "message": "This endpoint requires Premium subscription",
            "details": {
              "required_tier": "premium",
              "current_tier": "free",
              "upgrade_url": "https://aitradingpro.com/upgrade"
            }
          }
        }
        ```
        
        **429 Too Many Requests - Rate Limited**
        ```json
        {
          "success": false,
          "error": {
            "code": "RATE_LIMIT_EXCEEDED",
            "message": "Rate limit exceeded for your subscription tier",
            "details": {
              "limit": 10000,
              "reset_time": "2024-01-15T15:00:00Z",
              "retry_after": 3600
            }
          }
        }
        ```
        """)
    
    @staticmethod
    def show_endpoints():
        """API Endpoints documentation"""
        st.header("📡 API Endpoints")
        
        st.markdown("""
        ## Core Prediction Endpoints
        
        ### `POST /v1/predict`
        Get AI price predictions for any supported asset with comprehensive analytics.
        
        **Request Parameters:**
        ```json
        {
          "ticker": "BTCUSD",                    // Required: Asset symbol
          "timeframe": "1day",                   // Optional: 15min, 1hour, 4hour, 1day
          "models": ["advanced_transformer"],    // Optional: Specific models to use
          "include_trading_plan": true,          // Optional: Include trading levels
          "include_risk_metrics": true,          // Optional: Include risk analysis
          "include_forecast": true,              // Optional: Include 5-day forecast
          "include_technicals": true,            // Optional: Include technical analysis
          "include_alternative_data": true,      // Optional: Include sentiment data
          "confidence_threshold": 60             // Optional: Minimum confidence filter
        }
        ```
        
        **Response Structure:**
        ```json
        {
          "success": true,
          "data": {
            "ticker": "BTCUSD",
            "current_price": 43250.75,
            "predicted_price": 44180.30,
            "price_change_pct": 2.15,
            "confidence": 78.5,
            "timestamp": "2024-01-15T14:30:00Z",
            "trading_plan": { /* ... */ },
            "risk_metrics": { /* ... */ },
            "forecast_5_day": [44180.30, 44890.15, 45120.80],
            "technical_signals": { /* ... */ },
            "alternative_data": { /* ... */ }
          },
          "metadata": { /* ... */ }
        }
        ```
        
        ### `POST /v1/predict/batch`
        Get predictions for multiple assets in a single efficient request.
        
        **Request Parameters:**
        ```json
        {
          "requests": [
            {
              "ticker": "BTCUSD",
              "timeframe": "1day",
              "models": ["advanced_transformer", "xgboost"]
            },
            {
              "ticker": "ETHUSD", 
              "timeframe": "4hour",
              "models": ["cnn_lstm"]
            }
          ],
          "parallel": true,                      // Process requests in parallel
          "include_trading_plan": true,
          "include_risk_metrics": false
        }
        ```
        
        **Rate Limit Impact:** Batch requests count as number of individual requests.
        
        ### `GET /v1/models`
        Get information about available AI models and their characteristics.
        
        **Response:**
        ```json
        {
          "success": true,
          "data": {
            "models": [
              {
                "name": "advanced_transformer",
                "type": "neural_network", 
                "confidence": 0.85,
                "bias": 1.005,
                "strengths": ["long-term trends", "pattern recognition"],
                "best_for": ["multi-day forecasts", "trend analysis"]
              }
            ]
          }
        }
        ```
        
        ## Market Data Endpoints
        
        ### `GET /v1/market/price/{ticker}`
        Get current real-time market price for an asset.
        
        **Parameters:**
        - `ticker` (path): Asset symbol (e.g., BTCUSD, ^GSPC)
        
        **Response:**
        ```json
        {
          "success": true,
          "data": {
            "ticker": "BTCUSD",
            "price": 43250.75,
            "timestamp": "2024-01-15T14:30:00Z",
            "source": "real_time"
          }
        }
        ```
        
        ### `GET /v1/market/status`
        Get current global market status and trading hours.
        
        **Response:**
        ```json
        {
          "success": true,
          "data": {
            "is_market_open": true,
            "next_open": "2024-01-16T09:30:00Z",
            "next_close": "2024-01-15T16:00:00Z",
            "timezone": "America/New_York",
            "active_sessions": ["US", "EU"]
          }
        }
        ```
        
        ## Premium Analytics Endpoints
        
        ### `POST /v1/analytics/risk`
        Calculate comprehensive risk metrics for a trading strategy.
        
        **Parameters:**
        ```json
        {
          "ticker": "BTCUSD",
          "period": "3month",                    // 1month, 3month, 6month, 1year
          "confidence_level": 0.95,             // For VaR calculation
          "benchmark": "^GSPC"                   // Optional benchmark for comparison
        }
        ```
        
        **Response includes:** VaR, Expected Shortfall, Sharpe Ratio, Maximum Drawdown, Beta, Alpha
        
        ### `POST /v1/analytics/regime`
        Detect current market regime using advanced statistical models.
        
        **Parameters:**
        ```json
        {
          "ticker": "BTCUSD",
          "lookback_days": 60,                  // Analysis window
          "regime_types": 4                     // Number of regime states
        }
        ```
        
        ### `POST /v1/analytics/sentiment`
        Get comprehensive market sentiment analysis from multiple sources.
        
        **Parameters:**
        ```json
        {
          "ticker": "BTCUSD", 
          "sources": ["reddit", "twitter", "news"],
          "timeframe": "24h"                    // 1h, 6h, 24h, 7d
        }
        ```
        
        ## Backtesting Endpoints (Premium)
        
        ### `POST /v1/backtest`
        Run comprehensive strategy backtests with detailed performance metrics.
        
        **Parameters:**
        ```json
        {
          "ticker": "BTCUSD",
          "strategy": {
            "type": "ai_predictions",
            "models": ["advanced_transformer", "xgboost"],
            "confidence_threshold": 70,
            "position_size": 0.1,
            "stop_loss": 0.01,
            "take_profit": 0.02
          },
          "start_date": "2023-01-01",
          "end_date": "2023-12-31", 
          "initial_capital": 10000,
          "commission": 0.001,
          "slippage": 0.0005
        }
        ```
        
        **Response includes:** Total return, Sharpe ratio, Maximum drawdown, Win rate, Trade statistics
        
        ## Portfolio Endpoints (Premium)
        
        ### `POST /v1/portfolio/optimize`
        Optimize portfolio allocation using modern portfolio theory.
        
        **Parameters:**
        ```json
        {
          "tickers": ["BTCUSD", "ETHUSD", "^GSPC", "GC=F"],
          "objective": "max_sharpe",             // max_sharpe, min_risk, max_return
          "constraints": {
            "max_weight": 0.4,                  // Maximum weight per asset
            "min_weight": 0.05,                 // Minimum weight per asset
            "target_return": 0.12               // Target annual return (if applicable)
          },
          "lookback_period": "1year",
          "rebalance_frequency": "quarterly"
        }
        ```
        
        ### `POST /v1/portfolio/analysis`
        Analyze existing portfolio performance and risk characteristics.
        
        **Parameters:**
        ```json
        {
          "positions": [
            {"ticker": "BTCUSD", "weight": 0.3},
            {"ticker": "^GSPC", "weight": 0.7}
          ],
          "benchmark": "^GSPC",
          "analysis_period": "1year"
        }
        ```
        
        ## Webhook Endpoints (Premium)
        
        ### `POST /v1/webhooks/create`
        Create webhooks for real-time notifications when conditions are met.
        
        **Parameters:**
        ```json
        {
          "url": "https://your-server.com/webhook",
          "events": ["prediction_ready", "high_confidence_signal"],
          "filters": {
            "tickers": ["BTCUSD", "ETHUSD"],
            "min_confidence": 80,
            "max_confidence": 100
          },
          "secret": "your_webhook_secret"
        }
        ```
        
        ### `GET /v1/webhooks`
        List all active webhooks for your account.
        
        ### `DELETE /v1/webhooks/{webhook_id}`
        Delete a specific webhook.
        
        ## Historical Data Endpoints
        
        ### `GET /v1/historical/{ticker}`
        Get historical price data with technical indicators.
        
        **Parameters:**
        - `ticker` (path): Asset symbol
        - `period` (query): 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        - `interval` (query): 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        - `include_indicators` (query): boolean, include technical indicators
        
        **Response:**
        ```json
        {
          "success": true,
          "data": {
            "ticker": "BTCUSD",
            "period": "1mo",
            "interval": "1d",
            "data": [
              {
                "date": "2024-01-01",
                "open": 42000.00,
                "high": 43500.00,
                "low": 41800.00,
                "close": 43250.00,
                "volume": 1250000,
                "rsi": 65.4,
                "macd": 125.6,
                "bb_upper": 44000.0,
                "bb_middle": 43000.0,
                "bb_lower": 42000.0
              }
            ]
          }
        }
        ```
        
        ## Model Management Endpoints (Premium)
        
        ### `POST /v1/models/train`
        Trigger custom model training with your specified parameters.
        
        **Parameters:**
        ```json
        {
          "ticker": "BTCUSD",
          "models": ["advanced_transformer", "xgboost"],
          "training_period": "2year",
          "validation_method": "time_series_cv",
          "hyperparameters": {
            "advanced_transformer": {
              "d_model": 256,
              "nhead": 8,
              "num_layers": 6
            }
          }
        }
        ```
        
        ### `GET /v1/models/performance`
        Get detailed performance metrics for all models.
        
        **Response includes:** Cross-validation scores, bias analysis, confidence distributions
        
        ### `POST /v1/models/drift/detect`
        Check for model drift and recommend retraining.
        
        **Parameters:**
        ```json
        {
          "ticker": "BTCUSD",
          "reference_period": "3month",
          "detection_period": "1week"
        }
        ```
        
        ## Response Status Codes
        
        | Code | Meaning | Description |
        |------|---------|-------------|
        | 200 | OK | Request successful |
        | 201 | Created | Resource created successfully |
        | 400 | Bad Request | Invalid request parameters |
        | 401 | Unauthorized | Missing or invalid API key |
        | 403 | Forbidden | Insufficient permissions |
        | 404 | Not Found | Resource not found |
        | 429 | Too Many Requests | Rate limit exceeded |
        | 500 | Internal Server Error | Server error |
        | 503 | Service Unavailable | Temporary service disruption |
        
        ## Request/Response Headers
        
        **Required Request Headers:**
        ```
        Authorization: Bearer YOUR_API_KEY
        Content-Type: application/json
        User-Agent: YourApp/1.0
        ```
        
        **Important Response Headers:**
        ```
        X-RateLimit-Limit: 10000
        X-RateLimit-Remaining: 9850  
        X-RateLimit-Reset: 1642262400
        X-Request-ID: req_1234567890
        X-Processing-Time: 1250ms
        ```
        """)
    
    @staticmethod
    def show_examples():
        """API Examples"""
        st.header("📝 Code Examples")
        
        st.markdown("""
        ## Python Examples
        
        ### Complete Trading Bot Example
        ```python
        import requests
        import json
        import time
        from datetime import datetime
        
        class AITradingBot:
            def __init__(self, api_key, base_url="https://api.aitradingpro.com/v1"):
                self.api_key = api_key
                self.base_url = base_url
                self.session = requests.Session()
                self.session.headers.update({
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "AITradingBot/1.0"
                })
            
            def get_prediction(self, ticker, timeframe="1day", models=None):
                \"\"\"Get AI prediction with full analytics\"\"\"
                if models is None:
                    models = ["advanced_transformer", "xgboost", "cnn_lstm"]
                
                payload = {
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "models": models,
                    "include_trading_plan": True,
                    "include_risk_metrics": True,
                    "include_forecast": True,
                    "confidence_threshold": 70
                }
                
                try:
                    response = self.session.post(f"{self.base_url}/predict", json=payload)
                    response.raise_for_status()
                    return response.json()["data"]
                except requests.exceptions.RequestException as e:
                    print(f"API Error: {e}")
                    return None
            
            def get_market_sentiment(self, ticker):
                \"\"\"Get market sentiment analysis\"\"\"
                payload = {
                    "ticker": ticker,
                    "sources": ["reddit", "twitter", "news"],
                    "timeframe": "24h"
                }
                
                try:
                    response = self.session.post(f"{self.base_url}/analytics/sentiment", json=payload)
                    response.raise_for_status()
                    return response.json()["data"]
                except requests.exceptions.RequestException as e:
                    print(f"Sentiment API Error: {e}")
                    return None
            
            def analyze_opportunity(self, ticker):
                \"\"\"Complete opportunity analysis\"\"\"
                # Get prediction
                prediction = self.get_prediction(ticker)
                if not prediction:
                    return None
                
                # Get sentiment
                sentiment = self.get_market_sentiment(ticker)
                
                # Combine analysis
                opportunity = {
                    "ticker": ticker,
                    "prediction": prediction,
                    "sentiment": sentiment,
                    "timestamp": datetime.now().isoformat(),
                    "score": self.calculate_opportunity_score(prediction, sentiment)
                }
                
                return opportunity
            
            def calculate_opportunity_score(self, prediction, sentiment):
                \"\"\"Calculate opportunity score (0-100)\"\"\"
                score = 0
                
                # Base score from confidence
                score += prediction["confidence"] * 0.4
                
                # Direction alignment bonus
                pred_bullish = prediction["price_change_pct"] > 0
                sentiment_bullish = sentiment and sentiment.get("overall_score", 0) > 0
                if pred_bullish == sentiment_bullish:
                    score += 20
                
                # Risk adjustment
                risk_metrics = prediction.get("risk_metrics", {})
                sharpe = risk_metrics.get("sharpe_ratio", 0)
                if sharpe > 1.5:
                    score += 15
                elif sharpe > 1.0:
                    score += 10
                
                return min(100, max(0, score))
            
            def scan_opportunities(self, tickers, min_score=75):
                \"\"\"Scan multiple tickers for opportunities\"\"\"
                opportunities = []
                
                for ticker in tickers:
                    try:
                        opportunity = self.analyze_opportunity(ticker)
                        if opportunity and opportunity["score"] >= min_score:
                            opportunities.append(opportunity)
                        time.sleep(0.1)  # Rate limiting
                    except Exception as e:
                        print(f"Error analyzing {ticker}: {e}")
                        continue
                
                # Sort by opportunity score
                opportunities.sort(key=lambda x: x["score"], reverse=True)
                return opportunities
        
        # Usage Example
        if __name__ == "__main__":
            # Initialize bot
            bot = AITradingBot("your_api_key_here")
            
            # Define watchlist
            watchlist = ["BTCUSD", "ETHUSD", "^GSPC", "GC=F"]
            
            # Scan for opportunities
            opportunities = bot.scan_opportunities(watchlist, min_score=70)
            
            # Display results
            print("🔍 Top Trading Opportunities:")
            for opp in opportunities[:3]:  # Top 3
                pred = opp["prediction"]
                print(f"\\n📈 {opp['ticker']}")
                print(f"   Score: {opp['score']:.1f}/100")
                print(f"   Predicted Change: {pred['price_change_pct']:+.2f}%")
                print(f"   Confidence: {pred['confidence']:.1f}%")
                print(f"   Entry: ${pred['current_price']:.2f}")
                
                if 'trading_plan' in pred:
                    tp = pred['trading_plan']
                    print(f"   Strategy: {tp['strategy']}")
                    print(f"   Stop Loss: ${tp['stop_loss']:.2f}")
                    print(f"   Target: ${tp['target1']:.2f}")
        ```
        
        ### Batch Processing Example
        ```python
        def process_portfolio_batch(api_key, portfolio_tickers):
            \"\"\"Process multiple tickers efficiently\"\"\"
            
            # Prepare batch request
            requests_batch = []
            for ticker in portfolio_tickers:
                requests_batch.append({
                    "ticker": ticker,
                    "timeframe": "1day",
                    "models": ["advanced_transformer", "xgboost"],
                    "include_trading_plan": True
                })
            
            payload = {
                "requests": requests_batch,
                "parallel": True,
                "include_risk_metrics": True
            }
            
            # Make batch request
            response = requests.post(
                "https://api.aitradingpro.com/v1/predict/batch",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=payload
            )
            
            if response.status_code == 200:
                results = response.json()["data"]
                
                # Process results
                portfolio_analysis = {
                    "timestamp": datetime.now().isoformat(),
                    "total_assets": len(portfolio_tickers),
                    "bullish_signals": 0,
                    "bearish_signals": 0,
                    "high_confidence": 0,
                    "recommendations": []
                }
                
                for result in results["predictions"]:
                    ticker = result["ticker"]
                    change_pct = result["price_change_pct"]
                    confidence = result["confidence"]
                    
                    # Count signals
                    if change_pct > 0:
                        portfolio_analysis["bullish_signals"] += 1
                    else:
                        portfolio_analysis["bearish_signals"] += 1
                    
                    if confidence > 75:
                        portfolio_analysis["high_confidence"] += 1
                    
                    # Add recommendation
                    portfolio_analysis["recommendations"].append({
                        "ticker": ticker,
                        "action": "BUY" if change_pct > 1 else "SELL" if change_pct < -1 else "HOLD",
                        "confidence": confidence,
                        "expected_change": change_pct
                    })
                
                return portfolio_analysis
            else:
                print(f"Batch request failed: {response.status_code}")
                return None
        ```
        
        ### Risk Management Integration
        ```python
        def implement_risk_management(api_key, ticker, account_balance):
            \"\"\"Implement comprehensive risk management\"\"\"
            
            # Get prediction with risk metrics
            response = requests.post(
                "https://api.aitradingpro.com/v1/predict",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "ticker": ticker,
                    "include_risk_metrics": True,
                    "include_trading_plan": True
                }
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()["data"]
            
            # Risk analysis
            risk_metrics = data["risk_metrics"]
            trading_plan = data["trading_plan"]
            confidence = data["confidence"]
            
            # Calculate position size using Kelly Criterion
            win_rate = confidence / 100
            avg_win = abs(trading_plan["target1"] - trading_plan["entry_price"])
            avg_loss = abs(trading_plan["entry_price"] - trading_plan["stop_loss"])
            
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            else:
                kelly_fraction = 0.02  # Default 2%
            
            # Adjust for confidence and risk metrics
            confidence_multiplier = confidence / 100
            volatility_adjustment = min(1.0, 0.2 / risk_metrics.get("volatility", 0.2))
            
            final_position_size = kelly_fraction * confidence_multiplier * volatility_adjustment
            position_value = account_balance * final_position_size
            
            return {
                "ticker": ticker,
                "recommended_position_size": final_position_size,
                "position_value": position_value,
                "entry_price": trading_plan["entry_price"],
                "stop_loss": trading_plan["stop_loss"],
                "take_profit": trading_plan["target1"],
                "max_loss": position_value * (avg_loss / trading_plan["entry_price"]),
                "risk_metrics": risk_metrics,
                "kelly_fraction": kelly_fraction
            }
        ```
        
        ## JavaScript/TypeScript Examples
        
        ### React Component Integration
        ```typescript
        import React, { useState, useEffect } from 'react';
        import axios from 'axios';
        
        interface PredictionData {
          ticker: string;
          predicted_price: number;
          current_price: number;
          confidence: number;
          price_change_pct: number;
          trading_plan: {
            strategy: string;
            entry_price: number;
            stop_loss: number;
            target1: number;
          };
        }
        
        const TradingDashboard: React.FC = () => {
          const [predictions, setPredictions] = useState<PredictionData[]>([]);
          const [loading, setLoading] = useState(false);
          const [error, setError] = useState<string | null>(null);
          
          const apiKey = process.env.REACT_APP_ATP_API_KEY;
          
          const fetchPredictions = async (tickers: string[]) => {
            setLoading(true);
            setError(null);
            
            try {
              const requests = tickers.map(ticker => ({
                ticker,
                timeframe: '1day',
                models: ['advanced_transformer', 'xgboost'],
                include_trading_plan: true
              }));
              
              const response = await axios.post(
                'https://api.aitradingpro.com/v1/predict/batch',
                {
                  requests,
                  parallel: true
                },
                {
                  headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json'
                  }
                }
              );
              
              setPredictions(response.data.data.predictions);
            } catch (err) {
              setError('Failed to fetch predictions');
              console.error('API Error:', err);
            } finally {
              setLoading(false);
            }
          };
          
          useEffect(() => {
            const watchlist = ['BTCUSD', 'ETHUSD', '^GSPC'];
            fetchPredictions(watchlist);
            
            // Auto-refresh every 5 minutes
            const interval = setInterval(() => {
              fetchPredictions(watchlist);
            }, 5 * 60 * 1000);
            
            return () => clearInterval(interval);
          }, []);
          
          if (loading) return <div>Loading predictions...</div>;
          if (error) return <div>Error: {error}</div>;
          
          return (
            <div className="trading-dashboard">
              <h2>AI Trading Predictions</h2>
              <div className="predictions-grid">
                {predictions.map((pred) => (
                  <div key={pred.ticker} className="prediction-card">
                    <h3>{pred.ticker}</h3>
                    <div className="price-info">
                      <span>Current: ${pred.current_price.toFixed(2)}</span>
                      <span>Predicted: ${pred.predicted_price.toFixed(2)}</span>
                      <span className={pred.price_change_pct > 0 ? 'positive' : 'negative'}>
                        {pred.price_change_pct > 0 ? '+' : ''}{pred.price_change_pct.toFixed(2)}%
                      </span>
                    </div>
                    <div className="confidence">
                      Confidence: {pred.confidence.toFixed(1)}%
                    </div>
                    <div className="trading-plan">
                      <div>Strategy: {pred.trading_plan.strategy}</div>
                      <div>Entry: ${pred.trading_plan.entry_price.toFixed(2)}</div>
                      <div>Stop: ${pred.trading_plan.stop_loss.toFixed(2)}</div>
                      <div>Target: ${pred.trading_plan.target1.toFixed(2)}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        };
        
        export default TradingDashboard;
        ```
        
        ### Node.js Webhook Handler
        ```javascript
        const express = require('express');
        const crypto = require('crypto');
        const app = express();
        
        app.use(express.json());
        
        // Webhook endpoint
        app.post('/webhook/predictions', (req, res) => {
          const signature = req.headers['x-atp-signature'];
          const payload = JSON.stringify(req.body);
          const secret = process.env.ATP_WEBHOOK_SECRET;
          
          // Verify webhook signature
          const expectedSignature = crypto
            .createHmac('sha256', secret)
            .update(payload)
            .digest('hex');
          
          if (signature !== `sha256=${expectedSignature}`) {
            return res.status(401).send('Invalid signature');
          }
          
          // Process the webhook
          const { event, data } = req.body;
          
          switch (event) {
            case 'prediction_ready':
              handlePredictionReady(data);
              break;
            case 'high_confidence_signal':
              handleHighConfidenceSignal(data);
              break;
            default:
              console.log('Unknown event:', event);
          }
          
          res.status(200).send('OK');
        });
        
        function handlePredictionReady(data) {
          console.log('New prediction ready:', {
            ticker: data.ticker,
            predicted_change: data.price_change_pct,
            confidence: data.confidence
          });
          
          // Implement your trading logic here
          if (data.confidence > 80 && Math.abs(data.price_change_pct) > 2) {
            console.log('🚨 High-confidence trading opportunity detected!');
            // Send notification, execute trade, etc.
          }
        }
        
        function handleHighConfidenceSignal(data) {
          console.log('High confidence signal:', data);
          // Implement immediate action logic
        }
        
        app.listen(3000, () => {
          console.log('Webhook server running on port 3000');
        });
        ```
        
        ## Python Advanced Features
        
        ### Custom Strategy Backtesting
        ```python
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        def backtest_ai_strategy(api_key, ticker, start_date, end_date):
            \"\"\"Backtest AI prediction strategy\"\"\"
            
            # Configure backtest parameters
            backtest_config = {
                "ticker": ticker,
                "strategy": {
                    "type": "ai_predictions",
                    "models": ["advanced_transformer", "cnn_lstm", "xgboost"],
                    "confidence_threshold": 75,
                    "position_size": 0.1,  # 10% of capital per trade
                    "stop_loss": 0.01,     # 1% stop loss
                    "take_profit": 0.02,   # 2% take profit
                    "hold_period": 5       # Max 5 days
                },
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": 100000,
                "commission": 0.001,       # 0.1% commission
                "slippage": 0.0005        # 0.05% slippage
            }
            
            # Run backtest
            response = requests.post(
                "https://api.aitradingpro.com/v1/backtest",
                headers={"Authorization": f"Bearer {api_key}"},
                json=backtest_config
            )
            
            if response.status_code == 200:
                results = response.json()["data"]
                
                # Analyze results
                analysis = {
                    "total_return": results["total_return"],
                    "annual_return": results["annualized_return"],
                    "sharpe_ratio": results["sharpe_ratio"],
                    "max_drawdown": results["max_drawdown"],
                    "win_rate": results["win_rate"],
                    "profit_factor": results["profit_factor"],
                    "total_trades": results["total_trades"],
                    "avg_trade_duration": results.get("avg_trade_duration", "N/A")
                }
                
                print("📊 Backtest Results:")
                print(f"   Total Return: {analysis['total_return']:.2%}")
                print(f"   Annual Return: {analysis['annual_return']:.2%}")
                print(f"   Sharpe Ratio: {analysis['sharpe_ratio']:.2f}")
                print(f"   Max Drawdown: {analysis['max_drawdown']:.2%}")
                print(f"   Win Rate: {analysis['win_rate']:.1%}")
                print(f"   Total Trades: {analysis['total_trades']}")
                
                return analysis
            else:
                print(f"Backtest failed: {response.status_code}")
                return None
        
        # Usage
        results = backtest_ai_strategy(
            "your_api_key",
            "BTCUSD", 
            "2023-01-01",
            "2023-12-31"
        )
        ```
        
        ### Portfolio Optimization Example
        ```python
        def optimize_crypto_portfolio(api_key, budget=10000):
            \"\"\"Optimize a cryptocurrency portfolio\"\"\"
            
            crypto_tickers = ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD"]
            
            optimization_request = {
                "tickers": crypto_tickers,
                "objective": "max_sharpe",
                "constraints": {
                    "max_weight": 0.5,      # Max 50% in any single asset
                    "min_weight": 0.1,      # Min 10% in each asset
                    "target_volatility": 0.25  # Target 25% annual volatility
                },
                "lookback_period": "1year",
                "rebalance_frequency": "monthly"
            }
            
            response = requests.post(
                "https://api.aitradingpro.com/v1/portfolio/optimize",
                headers={"Authorization": f"Bearer {api_key}"},
                json=optimization_request
            )
            
            if response.status_code == 200:
                portfolio = response.json()["data"]
                
                print("🎯 Optimized Crypto Portfolio:")
                print(f"   Expected Return: {portfolio['expected_return']:.2%}")
                print(f"   Expected Volatility: {portfolio['expected_volatility']:.2%}")
                print(f"   Sharpe Ratio: {portfolio['sharpe_ratio']:.2f}")
                print("\\n   Allocation:")
                
                for ticker, weight in zip(portfolio["tickers"], portfolio["weights"]):
                    allocation = budget * weight
                    print(f"     {ticker}: {weight:.1%} (${allocation:,.0f})")
                
                return portfolio
            else:
                print(f"Portfolio optimization failed: {response.status_code}")
                return None
        ```
        """)
    
    @staticmethod
    def show_response_format():
        """Response format documentation"""
        st.header("📊 Response Format")
        
        st.markdown("""
        ## Standard Response Structure
        
        All API responses follow a consistent JSON format with standardized fields for reliability and ease of integration.
        
        ### Success Response Format
        ```json
        {
          "success": true,
          "data": {
            // Endpoint-specific data
          },
          "metadata": {
            "request_id": "req_1234567890abcdef",
            "processing_time_ms": 1250,
            "api_version": "v1.2.0",
            "timestamp": "2024-01-15T14:30:00Z",
            "rate_limit_remaining": 9950,
            "rate_limit_reset": 1642262400,
            "model_versions": {
              "advanced_transformer": "2.1.0",
              "xgboost": "1.7.3"
            }
          }
        }
        ```
        
        ### Error Response Format
        ```json
        {
          "success": false,
          "error": {
            "code": "VALIDATION_ERROR",
            "message": "Invalid ticker symbol provided",
            "details": {
              "field": "ticker",
              "provided_value": "INVALID_SYMBOL",
              "expected_format": "Valid asset symbol (e.g., BTCUSD, ^GSPC)",
              "supported_assets": [
                "BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD",
                "^GSPC", "^GDAXI", "^HSI", "^SPX",
                "GC=F", "CC=F", "NG=F", "KC=F", "SI=F", "HG=F",
                "USDJPY"
              ]
            }
          },
          "metadata": {
            "request_id": "err_1234567890abcdef",
            "processing_time_ms": 50,
            "api_version": "v1.2.0",
            "timestamp": "2024-01-15T14:30:00Z"
          }
        }
        ```
        
        ## Prediction Response Details
        
        ### Core Prediction Fields
        ```json
        {
          "ticker": "BTCUSD",
          "asset_type": "crypto",
          "current_price": 43250.75,
          "predicted_price": 44180.30,
          "price_change": 929.55,
          "price_change_pct": 2.15,
          "confidence": 78.5,
          "timestamp": "2024-01-15T14:30:00Z",
          "timeframe": "1day",
          "models_used": ["advanced_transformer", "xgboost", "cnn_lstm"],
          "prediction_horizon": "24h",
          "market_session": "US_REGULAR"
        }
        ```
        
        ### Trading Plan Structure
        ```json
        {
          "trading_plan": {
            "strategy": "BUY",
            "direction": "Long Position",
            "entry_price": 43250.75,
            "stop_loss": 42868.24,
            "target1": 43688.26,
            "target2": 44125.77,
            "risk_amount": 382.51,
            "reward1_amount": 437.51,
            "reward2_amount": 875.02,
            "risk_reward1": 1.14,
            "risk_reward2": 2.29,
            "position_size_pct": 2.5,
            "max_loss_pct": 0.88,
            "expected_duration": "1-3 days",
            "market_conditions": "favorable"
          }
        }
        ```
        
        ### Forecast Data
        ```json
        {
          "forecast_5_day": [44180.30, 44890.15, 45120.80, 44950.25, 45380.90],
          "forecast_metadata": {
            "method": "ensemble_average",
            "confidence_intervals": {
              "day_1": {"lower": 43500, "upper": 44900},
              "day_2": {"lower": 44100, "upper": 45700},
              "day_3": {"lower": 44300, "upper": 46000},
              "day_4": {"lower": 44000, "upper": 45900},
              "day_5": {"lower": 44500, "upper": 46300}
            },
            "trend_direction": "bullish",
            "volatility_forecast": [0.025, 0.028, 0.032, 0.029, 0.031]
          }
        }
        ```
        
        ### Risk Metrics Structure (Premium)
        ```json
        {
          "risk_metrics": {
            "var_95": -0.0287,
            "var_99": -0.0421,
            "expected_shortfall": -0.0356,
            "sharpe_ratio": 1.47,
            "sortino_ratio": 2.13,
            "calmar_ratio": 0.89,
            "max_drawdown": -0.1456,
            "volatility": 0.2341,
            "skewness": -0.123,
            "kurtosis": 2.876,
            "beta": 1.23,
            "alpha": 0.0456,
            "tracking_error": 0.0234,
            "information_ratio": 1.95,
            "downside_deviation": 0.1567,
            "upside_capture": 0.987,
            "downside_capture": 0.823
          }
        }
        ```
        
        ### Technical Analysis Structure
        ```json
        {
          "technical_analysis": {
            "rsi": {
              "value": 65.4,
              "signal": "neutral",
              "overbought_threshold": 70,
              "oversold_threshold": 30
            },
            "macd": {
              "value": 125.6,
              "signal_line": 118.9,
              "histogram": 6.7,
              "signal": "bullish"
            },
            "bollinger_bands": {
              "upper": 44500.0,
              "middle": 43250.0,
              "lower": 42000.0,
              "position": 0.67,
              "signal": "neutral"
            },
            "moving_averages": {
              "sma_20": 42950.5,
              "sma_50": 42100.3,
              "ema_12": 43150.2,
              "ema_26": 42800.1,
              "golden_cross": false,
              "death_cross": false
            },
            "momentum": {
              "roc_10": 0.0234,
              "momentum_14": 1.0456,
              "williams_r": -23.4,
              "stochastic_k": 67.8,
              "stochastic_d": 65.2
            },
            "volume": {
              "obv": 12345678,
              "volume_sma": 2341567,
              "volume_ratio": 1.23,
              "accumulation_distribution": 567890
            },
            "support_resistance": {
              "support_levels": [41800, 42200, 42600],
              "resistance_levels": [44000, 44500, 45000],
              "pivot_point": 43250,
              "fibonacci_levels": {
                "23.6": 42850,
                "38.2": 42450,
                "50.0": 42100,
                "61.8": 41750,
                "78.6": 41400
              }
            }
          }
        }
        ```
        
        ### Alternative Data Structure (Premium)
        ```json
        {
          "alternative_data": {
            "sentiment": {
              "reddit_sentiment": 0.234,
              "twitter_sentiment": 0.156,
              "news_sentiment": 0.089,
              "overall_sentiment": 0.167,
              "sentiment_trend": "improving",
              "volume_weighted_sentiment": 0.203
            },
            "economic_indicators": {
              "dgs10": 4.25,
              "fedfunds": 5.25,
              "unrate": 3.8,
              "cpiaucsl": 287.5,
              "gdpc1": 21456.7,
              "dexuseu": 1.0823,
              "dcoilwtico": 78.45,
              "goldamgbd228nlbm": 1956.7,
              "m2sl": 20876.4,
              "umcsent": 78.9
            },
            "options_flow": {
              "put_call_ratio": 0.87,
              "implied_volatility": 0.34,
              "gamma_exposure": -1234567890,
              "dark_pool_index": 0.45,
              "max_pain": 43000,
              "options_volume": 150000,
              "put_volume": 65000,
              "call_volume": 85000
            },
            "market_microstructure": {
              "bid_ask_spread": 0.25,
              "market_impact": 0.00012,
              "price_improvement": 0.05,
              "effective_spread": 0.18,
              "realized_spread": 0.12,
              "adverse_selection": 0.06
            }
          }
        }
        ```
        
        ### Model Ensemble Analysis (Premium)
        ```json
        {
          "ensemble_analysis": {
            "advanced_transformer": {
              "prediction": 44285.6,
              "confidence": 85.2,
              "weight": 0.25,
              "model_type": "neural_network",
              "bias": "slightly_bullish",
              "features_used": 67,
              "training_r2": 0.847
            },
            "cnn_lstm": {
              "prediction": 44156.3,
              "confidence": 78.9,
              "weight": 0.22,
              "model_type": "neural_network",
              "bias": "neutral",
              "features_used": 45,
              "training_r2": 0.789
            },
            "xgboost": {
              "prediction": 44089.7,
              "confidence": 72.4,
              "weight": 0.18,
              "model_type": "tree_based",
              "bias": "slightly_bearish",
              "features_used": 89,
              "training_r2": 0.756
            }
          },
          "voting_results": {
            "unanimous_direction": true,
            "consensus_strength": 0.89,
            "prediction_variance": 234.5,
            "outlier_models": [],
            "agreement_score": 94.2
          }
        }
        ```
        
        ### Model Explanations Structure (Premium)
        ```json
        {
          "explanations": {
            "shap_values": {
              "top_positive_features": [
                {"feature": "close_price", "value": 0.234},
                {"feature": "rsi", "value": 0.156},
                {"feature": "volume_ratio", "value": 0.089}
              ],
              "top_negative_features": [
                {"feature": "macd_histogram", "value": -0.067},
                {"feature": "bb_position", "value": -0.045}
              ],
              "base_value": 43250.0,
              "expected_value": 44180.3
            },
            "feature_importance": {
              "close_price": 0.234,
              "volume": 0.156,
              "rsi": 0.123,
              "macd": 0.098,
              "bb_position": 0.087,
              "sma_20": 0.076,
              "ema_12": 0.065,
              "obv": 0.054,
              "atr": 0.043,
              "roc_10": 0.032
            },
            "permutation_importance": {
              "close_price": 0.045,
              "volume": 0.032,
              "rsi": 0.028,
              "macd": 0.025,
              "bb_position": 0.021
            },
            "model_specific_insights": {
              "dominant_patterns": ["upward_trend", "volume_confirmation"],
              "key_drivers": ["technical_momentum", "market_sentiment"],
              "risk_factors": ["volatility_spike", "sentiment_reversal"]
            }
          }
        }
        ```
        
        ### Batch Response Format
        ```json
        {
          "success": true,
          "data": {
            "batch_id": "batch_1234567890",
            "total_requests": 5,
            "successful_predictions": 5,
            "failed_predictions": 0,
            "processing_time_ms": 2340,
            "predictions": [
              {
                "request_index": 0,
                "ticker": "BTCUSD",
                "success": true,
                "data": { /* full prediction object */ }
              },
              {
                "request_index": 1,
                "ticker": "ETHUSD",
                "success": true,
                "data": { /* full prediction object */ }
              }
            ],
            "summary": {
              "bullish_signals": 3,
              "bearish_signals": 2,
              "high_confidence_signals": 4,
              "average_confidence": 76.8
            }
          }
        }
        ```
        
        ## Common Error Codes and Details
        
        ### Validation Errors (400)
        ```json
        {
          "error": {
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": {
              "errors": [
                {
                  "field": "ticker",
                  "message": "Invalid ticker symbol",
                  "provided": "INVALID",
                  "expected": "Valid asset symbol from supported list"
                },
                {
                  "field": "timeframe", 
                  "message": "Invalid timeframe",
                  "provided": "5min",
                  "expected": "One of: 15min, 1hour, 4hour, 1day"
                }
              ]
            }
          }
        }
        ```
        
        ### Rate Limit Errors (429)
        ```json
        {
          "error": {
            "code": "RATE_LIMIT_EXCEEDED",
            "message": "API rate limit exceeded",
            "details": {
              "current_usage": 10000,
              "limit": 10000,
              "reset_time": "2024-01-15T15:00:00Z",
              "retry_after": 3600,
              "upgrade_available": true,
              "upgrade_url": "https://aitradingpro.com/upgrade"
            }
          }
        }
        ```
        
        ### Authentication Errors (401/403)
        ```json
        {
          "error": {
            "code": "INSUFFICIENT_PERMISSIONS",
            "message": "Premium subscription required",
            "details": {
              "required_tier": "premium",
              "current_tier": "free",
              "feature": "advanced_models",
              "upgrade_url": "https://aitradingpro.com/upgrade"
            }
          }
        }
        ```
        
        ## Response Headers
        
        Every API response includes important metadata in the headers:
        
        ```
        X-RateLimit-Limit: 10000
        X-RateLimit-Remaining: 9850
        X-RateLimit-Reset: 1642262400
        X-RateLimit-Tier: premium
        X-Request-ID: req_1234567890abcdef
        X-Processing-Time: 1250ms
        X-Model-Version: 2.1.0
        X-API-Version: v1.2.0
        X-Cache-Status: miss
        X-Region: us-east-1
        ```
        
        ## Data Types and Formats
        
        ### Numeric Precision
        - **Prices**: Up to 8 decimal places for precision
        - **Percentages**: Up to 4 decimal places (e.g., 2.1534%)
        - **Confidence**: Always between 45.0 and 88.0
        - **Timestamps**: ISO 8601 format (UTC)
        
        ### Null Values
        - Missing optional fields are omitted rather than null
        - Failed calculations return reasonable defaults
        - Arrays are never null (empty arrays instead)
        
        ### Consistency Guarantees
        - Prediction timestamps match across all fields
        - Price calculations are mathematically consistent
        - Confidence scores align with historical performance
        - All monetary values use the same precision
        """)
    
    @staticmethod
    def show_rate_limits():
        """Rate limits documentation"""
        st.header("⚠️ Rate Limits & Usage Management")
        
        st.markdown("""
        ## Rate Limit Tiers
        
        Our API implements sophisticated rate limiting to ensure fair usage and optimal performance for all users.
        
        ### Free Tier Limits
        - **Hourly Limit**: 100 requests per hour
        - **Daily Predictions**: 10 predictions per day (same as web interface)
        - **Burst Capacity**: Up to 20 requests in 5 minutes
        - **Concurrent Requests**: Maximum 2 simultaneous requests
        - **Features**: Basic models only (XGBoost, Scikit-learn Ensemble)
        
        ### Premium Tier Limits  
        - **Hourly Limit**: 10,000 requests per hour
        - **Daily Predictions**: Unlimited
        - **Burst Capacity**: Up to 100 requests in 5 minutes
        - **Concurrent Requests**: Maximum 10 simultaneous requests
        - **Features**: All models and premium analytics
        
        ### Enterprise Tier
        - **Custom Limits**: Negotiated based on needs
        - **Dedicated Infrastructure**: Isolated resources
        - **SLA Guarantees**: 99.9% uptime commitment
        - **Priority Support**: 24/7 technical support
        - **Custom Integrations**: Dedicated integration support
        
        ## Rate Limit Headers
        
        Every API response includes comprehensive rate limit information:
        
        ```http
        X-RateLimit-Limit: 10000
        X-RateLimit-Remaining: 9850
        X-RateLimit-Reset: 1642262400
        X-RateLimit-Tier: premium
        X-RateLimit-Window: 3600
        X-RateLimit-Burst-Limit: 100
        X-RateLimit-Burst-Remaining: 95
        X-RateLimit-Retry-After: 3600
        ```
        
        ### Header Explanations
        - **X-RateLimit-Limit**: Total requests allowed in current window
        - **X-RateLimit-Remaining**: Requests remaining in current window
        - **X-RateLimit-Reset**: Unix timestamp when limit resets
        - **X-RateLimit-Window**: Rate limit window in seconds (3600 = 1 hour)
        - **X-RateLimit-Burst-Limit**: Burst capacity limit
        - **X-RateLimit-Burst-Remaining**: Burst requests remaining
        - **X-RateLimit-Retry-After**: Seconds to wait before retrying (when limited)
        
        ## Rate Limiting Strategies
        
        ### Smart Rate Limiting Implementation
        ```python
        import time
        import requests
        from datetime import datetime, timedelta
        
        class RateLimitedClient:
            def __init__(self, api_key, tier="premium"):
                self.api_key = api_key
                self.base_url = "https://api.aitradingpro.com/v1"
                self.session = requests.Session()
                self.session.headers.update({
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                })
                
                # Tier-specific settings
                if tier == "premium":
                    self.requests_per_hour = 10000
                    self.burst_limit = 100
                    self.concurrent_limit = 10
                else:
                    self.requests_per_hour = 100
                    self.burst_limit = 20
                    self.concurrent_limit = 2
                
                # Rate limiting state
                self.requests_made = 0
                self.window_start = datetime.now()
                self.last_request_time = 0
                self.burst_requests = 0
                self.burst_window_start = datetime.now()
                
            def make_request(self, endpoint, method="GET", **kwargs):
                \"\"\"Make rate-limited request with automatic retry\"\"\"
                
                # Check and handle rate limits
                self._enforce_rate_limits()
                
                # Make the request
                url = f"{self.base_url}/{endpoint.lstrip('/')}"
                
                try:
                    if method.upper() == "POST":
                        response = self.session.post(url, **kwargs)
                    else:
                        response = self.session.get(url, **kwargs)
                    
                    # Update rate limiting counters
                    self._update_counters(response)
                    
                    # Handle rate limit responses
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('X-RateLimit-Retry-After', 60))
                        print(f"Rate limited. Waiting {retry_after} seconds...")
                        time.sleep(retry_after)
                        return self.make_request(endpoint, method, **kwargs)
                    
                    response.raise_for_status()
                    return response.json()
                    
                except requests.exceptions.RequestException as e:
                    print(f"Request failed: {e}")
                    return None
            
            def _enforce_rate_limits(self):
                \"\"\"Enforce rate limits before making request\"\"\"
                now = datetime.now()
                
                # Reset hourly window if needed
                if (now - self.window_start).seconds >= 3600:
                    self.requests_made = 0
                    self.window_start = now
                
                # Reset burst window if needed (5 minutes)
                if (now - self.burst_window_start).seconds >= 300:
                    self.burst_requests = 0
                    self.burst_window_start = now
                
                # Check hourly limit
                if self.requests_made >= self.requests_per_hour:
                    sleep_time = 3600 - (now - self.window_start).seconds
                    print(f"Hourly limit reached. Waiting {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    self.requests_made = 0
                    self.window_start = datetime.now()
                
                # Check burst limit
                if self.burst_requests >= self.burst_limit:
                    sleep_time = 300 - (now - self.burst_window_start).seconds
                    if sleep_time > 0:
                        print(f"Burst limit reached. Waiting {sleep_time} seconds...")
                        time.sleep(sleep_time)
                        self.burst_requests = 0
                        self.burst_window_start = datetime.now()
                
                # Minimum interval between requests
                min_interval = 3600 / self.requests_per_hour
                elapsed = time.time() - self.last_request_time
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
            
            def _update_counters(self, response):
                \"\"\"Update rate limiting counters from response\"\"\"
                self.requests_made += 1
                self.burst_requests += 1
                self.last_request_time = time.time()
                
                # Update from response headers if available
                if 'X-RateLimit-Remaining' in response.headers:
                    remaining = int(response.headers['X-RateLimit-Remaining'])
                    limit = int(response.headers.get('X-RateLimit-Limit', self.requests_per_hour))
                    self.requests_made = limit - remaining
        
        # Usage example
        client = RateLimitedClient("your_api_key", "premium")
        
        # Safe batch processing
        tickers = ["BTCUSD", "ETHUSD", "^GSPC", "GC=F"]
        for ticker in tickers:
            result = client.make_request("predict", "POST", json={"ticker": ticker})
            if result:
                print(f"Prediction for {ticker}: {result['data']['predicted_price']}")
        ```
        
        ### Batch Request Optimization
        ```python
        def optimize_batch_requests(client, tickers, batch_size=10):
            \"\"\"Optimize multiple requests using batching\"\"\"
            
            results = []
            
            # Process in batches to minimize API calls
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i + batch_size]
                
                # Create batch request
                batch_requests = [
                    {"ticker": ticker, "timeframe": "1day"} 
                    for ticker in batch
                ]
                
                # Single API call for entire batch
                batch_result = client.make_request(
                    "predict/batch",
                    "POST",
                    json={
                        "requests": batch_requests,
                        "parallel": True
                    }
                )
                
                if batch_result and batch_result["success"]:
                    results.extend(batch_result["data"]["predictions"])
                
                # Small delay between batches
                time.sleep(0.1)
            
            return results
        ```
        
        ## Error Handling and Retry Logic
        
        ### Exponential Backoff Implementation
        ```python
        import random
        
        def exponential_backoff_request(client, endpoint, max_retries=5, **kwargs):
            \"\"\"Make request with exponential backoff on rate limit errors\"\"\"
            
            for attempt in range(max_retries):
                try:
                    response = client.session.request(**kwargs)
                    
                    if response.status_code == 429:
                        # Rate limited - calculate backoff
                        retry_after = int(response.headers.get('X-RateLimit-Retry-After', 60))
                        backoff_time = min(retry_after, (2 ** attempt) + random.uniform(0, 1))
                        
                        print(f"Rate limited. Attempt {attempt + 1}/{max_retries}. "
                              f"Backing off for {backoff_time:.2f} seconds...")
                        
                        time.sleep(backoff_time)
                        continue
                    
                    response.raise_for_status()
                    return response.json()
                    
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise e
                    
                    backoff_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Request failed. Retrying in {backoff_time:.2f} seconds...")
                    time.sleep(backoff_time)
            
            raise Exception(f"Max retries ({max_retries}) exceeded")
        ```
        
        ## Usage Monitoring and Analytics
        
        ### Track API Usage
        ```python
        class UsageTracker:
            def __init__(self):
                self.usage_stats = {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "rate_limited_requests": 0,
                    "average_response_time": 0,
                    "endpoints_used": {},
                    "daily_usage": {}
                }
            
            def track_request(self, endpoint, response_time, status_code):
                \"\"\"Track individual request metrics\"\"\"
                today = datetime.now().date().isoformat()
                
                # Update totals
                self.usage_stats["total_requests"] += 1
                
                if status_code == 200:
                    self.usage_stats["successful_requests"] += 1
                elif status_code == 429:
                    self.usage_stats["rate_limited_requests"] += 1
                else:
                    self.usage_stats["failed_requests"] += 1
                
                # Update average response time
                total_requests = self.usage_stats["total_requests"]
                current_avg = self.usage_stats["average_response_time"]
                self.usage_stats["average_response_time"] = (
                    (current_avg * (total_requests - 1) + response_time) / total_requests
                )
                
                # Track endpoint usage
                if endpoint not in self.usage_stats["endpoints_used"]:
                    self.usage_stats["endpoints_used"][endpoint] = 0
                self.usage_stats["endpoints_used"][endpoint] += 1
                
                # Track daily usage
                if today not in self.usage_stats["daily_usage"]:
                    self.usage_stats["daily_usage"][today] = 0
                self.usage_stats["daily_usage"][today] += 1
            
            def get_usage_report(self):
                \"\"\"Generate usage report\"\"\"
                stats = self.usage_stats
                
                success_rate = (stats["successful_requests"] / stats["total_requests"] * 100 
                               if stats["total_requests"] > 0 else 0)
                
                rate_limit_rate = (stats["rate_limited_requests"] / stats["total_requests"] * 100
                                  if stats["total_requests"] > 0 else 0)
                
                report = f\"\"\"
                📊 API Usage Report
                ==================
                Total Requests: {stats['total_requests']}
                Success Rate: {success_rate:.1f}%
                Rate Limited: {rate_limit_rate:.1f}%
                Avg Response Time: {stats['average_response_time']:.0f}ms
                
                Top Endpoints:
                \"\"\"
                
                # Sort endpoints by usage
                sorted_endpoints = sorted(
                    stats["endpoints_used"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for endpoint, count in sorted_endpoints[:5]:
                    report += f\"  {endpoint}: {count} requests\\n\"
                
                return report
        ```
        
        ## Best Practices for Rate Limit Management
        
        ### Production-Ready Implementation
        ```python
        class ProductionAPIClient:
            def __init__(self, api_key, tier="premium"):
                self.client = RateLimitedClient(api_key, tier)
                self.usage_tracker = UsageTracker()
                self.cache = {}
                self.cache_ttl = 300  # 5 minutes
            
            def get_prediction_with_cache(self, ticker, timeframe="1day"):
                \"\"\"Get prediction with intelligent caching\"\"\"
                cache_key = f\"{ticker}_{timeframe}\"
                now = time.time()
                
                # Check cache first
                if cache_key in self.cache:
                    cached_data, timestamp = self.cache[cache_key]
                    if now - timestamp < self.cache_ttl:
                        print(f\"Cache hit for {ticker}\")
                        return cached_data
                
                # Make API request
                start_time = time.time()
                result = self.client.make_request(
                    \"predict\",
                    \"POST\",
                    json={\"ticker\": ticker, \"timeframe\": timeframe}
                )
                response_time = (time.time() - start_time) * 1000
                
                # Track usage
                status_code = 200 if result else 500
                self.usage_tracker.track_request(\"predict\", response_time, status_code)
                
                # Update cache
                if result:
                    self.cache[cache_key] = (result, now)
                
                return result
            
            def batch_predictions_optimized(self, tickers, max_batch_size=10):
                \"\"\"Optimized batch processing with intelligent grouping\"\"\"
                
                # Group tickers to minimize API calls
                batches = [tickers[i:i + max_batch_size] 
                          for i in range(0, len(tickers), max_batch_size)]
                
                all_results = []
                
                for batch in batches:
                    # Check cache for batch items
                    cached_results = []
                    uncached_tickers = []
                    
                    for ticker in batch:
                        cached_result = self.get_cached_result(ticker)
                        if cached_result:
                            cached_results.append(cached_result)
                        else:
                            uncached_tickers.append(ticker)
                    
                    # Only request uncached items
                    if uncached_tickers:
                        batch_requests = [
                            {\"ticker\": ticker, \"timeframe\": \"1day\"}
                            for ticker in uncached_tickers
                        ]
                        
                        start_time = time.time()
                        batch_result = self.client.make_request(
                            \"predict/batch\",
                            \"POST\",
                            json={\"requests\": batch_requests, \"parallel\": True}
                        )
                        response_time = (time.time() - start_time) * 1000
                        
                        # Track usage
                        status_code = 200 if batch_result else 500
                        self.usage_tracker.track_request(\"predict/batch\", response_time, status_code)
                        
                        if batch_result and batch_result[\"success\"]:
                            # Cache new results
                            for prediction in batch_result[\"data\"][\"predictions\"]:
                                cache_key = f\"{prediction['ticker']}_1day\"
                                self.cache[cache_key] = (prediction, time.time())
                            
                            all_results.extend(batch_result[\"data\"][\"predictions\"])
                    
                    # Add cached results
                    all_results.extend(cached_results)
                    
                    # Rate limiting delay between batches
                    if len(batches) > 1:
                        time.sleep(0.1)
                
                return all_results
            
            def get_cached_result(self, ticker, timeframe=\"1day\"):
                \"\"\"Get cached result if available and fresh\"\"\"
                cache_key = f\"{ticker}_{timeframe}\"
                if cache_key in self.cache:
                    cached_data, timestamp = self.cache[cache_key]
                    if time.time() - timestamp < self.cache_ttl:
                        return cached_data
                return None
            
            def cleanup_cache(self):
                \"\"\"Remove expired cache entries\"\"\"
                now = time.time()
                expired_keys = [
                    key for key, (_, timestamp) in self.cache.items()
                    if now - timestamp > self.cache_ttl
                ]
                for key in expired_keys:
                    del self.cache[key]
        ```
        
        ## Monitoring and Alerting
        
        ### Set up monitoring for rate limit violations:
        
        1. **Track Rate Limit Headers**: Monitor remaining requests
        2. **Alert on High Usage**: Warn when approaching limits  
        3. **Performance Monitoring**: Track response times
        4. **Error Rate Monitoring**: Alert on high error rates
        5. **Usage Analytics**: Understand usage patterns
        
        ### Recommended Monitoring Thresholds:
        - **Warning**: 80% of rate limit used
        - **Critical**: 95% of rate limit used
        - **Response Time**: Alert if >2 seconds
        - **Error Rate**: Alert if >5% errors
        """)

# =============================================================================
# SUPPORT SYSTEM (Placeholder for Integration with fixedui.py)
# =============================================================================

class SupportSystem:
    """Support system for AI Trading Professional (placeholder)."""
    
    @staticmethod
    def create_support_page():
        """This function can display a support or help page for the app."""
        st.title("Support & Help")
        st.write("If you need assistance, please consult the documentation or contact our support team.")


# =============================================================================
# FOOTER AND NAVIGATION ENHANCEMENTS (Placeholder for Integration with fixedui.py)
# =============================================================================

def create_enhanced_footer():
    """
    Creates an enhanced footer that can be used in the main app (fixedui.py).
    """
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.9rem; text-align:center; color: #888;">
            <p><strong>AI Trading Professional</strong> © 2025. All rights reserved.</p>
            <p>
                <a href="#documentation-page" style="margin-right:15px;">Documentation</a>
                <a href="#api-documentation" style="margin-right:15px;">API</a>
                <a href="#support-page">Support</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def handle_navigation():
    """
    Placeholder function to handle navigation logic that might be called from fixedui.py.
    """
    st.write("Navigation handler not yet implemented.")