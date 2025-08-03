"""
Timeframe Management System for AI Trading Professional
Handles multi-timeframe data and price calculations
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import necessary components from backend
try:
    from enhprog import (
        FMPDataProvider, 
        get_asset_type, 
        get_reasonable_price_range,
        calculate_rsi, 
        calculate_macd, 
        calculate_bollinger_bands,
        FMP_API_KEY
    )
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    logger.warning("Backend not available, using standalone mode")
    FMP_API_KEY = None

# =============================================================================
# TIMEFRAME CONFIGURATION
# =============================================================================

TIMEFRAME_CONFIG = {
    '15min': {
        'interval': '15min',
        'period_minutes': 15,
        'candles_per_day': 96,  # 24 * 60 / 15
        'lookback_days': 5,
        'max_change_per_period': {
            'crypto': 0.02,      # 2% per 15min
            'forex': 0.003,      # 0.3% per 15min
            'commodity': 0.008,  # 0.8% per 15min
            'index': 0.005,      # 0.5% per 15min
            'stock': 0.01        # 1% per 15min
        },
        'volatility_multiplier': 1.5,
        'update_frequency': 15  # seconds
    },
    '1hour': {
        'interval': '1h',
        'period_minutes': 60,
        'candles_per_day': 24,
        'lookback_days': 20,
        'max_change_per_period': {
            'crypto': 0.05,      # 5% per hour
            'forex': 0.008,      # 0.8% per hour
            'commodity': 0.02,   # 2% per hour
            'index': 0.015,      # 1.5% per hour
            'stock': 0.025       # 2.5% per hour
        },
        'volatility_multiplier': 1.2,
        'update_frequency': 60  # seconds
    },
    '4hour': {
        'interval': '4h',
        'period_minutes': 240,
        'candles_per_day': 6,
        'lookback_days': 60,
        'max_change_per_period': {
            'crypto': 0.12,      # 12% per 4 hours
            'forex': 0.02,       # 2% per 4 hours
            'commodity': 0.06,   # 6% per 4 hours
            'index': 0.04,       # 4% per 4 hours
            'stock': 0.08        # 8% per 4 hours
        },
        'volatility_multiplier': 1.0,
        'update_frequency': 300  # seconds
    },
    '1day': {
        'interval': '1d',
        'period_minutes': 1440,
        'candles_per_day': 1,
        'lookback_days': 365,
        'max_change_per_period': {
            'crypto': 0.20,      # 20% per day
            'forex': 0.05,       # 5% per day
            'commodity': 0.10,   # 10% per day
            'index': 0.08,       # 8% per day
            'stock': 0.15        # 15% per day
        },
        'volatility_multiplier': 0.8,
        'update_frequency': 900  # seconds
    }
}

# =============================================================================
# PRICE CALCULATION ENGINE
# =============================================================================

class TimeframePriceCalculator:
    """Calculate prices based on timeframe with realistic movements"""
    
    def __init__(self):
        self.price_history = defaultdict(lambda: defaultdict(list))
        self.last_prices = defaultdict(dict)
        self.volatility_state = defaultdict(lambda: {'current': 1.0, 'target': 1.0})
        
    def calculate_timeframe_price(self, ticker: str, base_price: float, 
                                 timeframe: str, current_time: datetime = None) -> float:
        """
        Calculate price for specific timeframe with realistic movements
        """
        if current_time is None:
            current_time = datetime.now()
            
        asset_type = self._get_asset_type(ticker)
        config = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG['1day'])
        
        # Get last known price for this timeframe
        last_price = self.last_prices.get(ticker, {}).get(timeframe, base_price)
        
        # Calculate time-based volatility
        hour = current_time.hour
        volatility_adjustment = self._get_volatility_adjustment(asset_type, hour, timeframe)
        
        # Get maximum change for this period
        max_change = config['max_change_per_period'].get(asset_type, 0.02)
        
        # Apply volatility state
        volatility_state = self.volatility_state[ticker]
        volatility_state['current'] = 0.9 * volatility_state['current'] + 0.1 * volatility_state['target']
        
        # Calculate price movement
        if asset_type == 'crypto':
            # More volatile, trending movements
            trend = self._calculate_trend(ticker, timeframe)
            noise = np.random.normal(0, max_change * 0.3)
            change = trend * max_change * 0.7 + noise
        elif asset_type == 'forex':
            # Mean-reverting with small movements
            mean_reversion = self._calculate_mean_reversion(ticker, last_price, base_price)
            noise = np.random.normal(0, max_change * 0.2)
            change = mean_reversion * max_change * 0.5 + noise
        elif asset_type == 'commodity':
            # Supply/demand driven with cycles
            cycle = self._calculate_cycle(timeframe, current_time)
            noise = np.random.normal(0, max_change * 0.25)
            change = cycle * max_change * 0.6 + noise
        else:  # stock/index
            # Market hours affect volatility
            market_factor = self._get_market_hours_factor(current_time)
            noise = np.random.normal(0, max_change * 0.3)
            change = noise * market_factor
            
        # Apply volatility adjustments
        change *= volatility_adjustment * volatility_state['current']
        change *= config['volatility_multiplier']
        
        # Ensure change is within bounds
        change = np.clip(change, -max_change, max_change)
        
        # Calculate new price
        new_price = last_price * (1 + change)
        
        # Ensure price stays within reasonable bounds
        min_price, max_price = self._get_price_bounds(ticker)
        new_price = np.clip(new_price, min_price * 0.5, max_price * 2.0)
        
        # Update history
        self.last_prices[ticker][timeframe] = new_price
        self.price_history[ticker][timeframe].append({
            'time': current_time,
            'price': new_price,
            'change': change
        })
        
        # Maintain history size
        if len(self.price_history[ticker][timeframe]) > 1000:
            self.price_history[ticker][timeframe] = self.price_history[ticker][timeframe][-500:]
            
        return new_price
    
    def _get_asset_type(self, ticker: str) -> str:
        """Get asset type with fallback"""
        if BACKEND_AVAILABLE:
            return get_asset_type(ticker)
        else:
            # Fallback logic
            if ticker.startswith('^'):
                return 'index'
            elif '=F' in ticker:
                return 'commodity'
            elif 'USD' in ticker and len(ticker) == 6:
                if ticker in ['BTCUSD', 'ETHUSD', 'SOLUSD', 'BNBUSD']:
                    return 'crypto'
                else:
                    return 'forex'
            else:
                return 'stock'
    
    def _get_price_bounds(self, ticker: str) -> Tuple[float, float]:
        """Get price bounds with fallback"""
        if BACKEND_AVAILABLE:
            return get_reasonable_price_range(ticker)
        else:
            # Fallback price ranges
            asset_type = self._get_asset_type(ticker)
            ranges = {
                'crypto': (100, 100000),
                'forex': (0.5, 2.0),
                'commodity': (10, 500),
                'index': (1000, 50000),
                'stock': (10, 1000)
            }
            return ranges.get(asset_type, (10, 1000))
    
    def _get_volatility_adjustment(self, asset_type: str, hour: int, timeframe: str) -> float:
        """Get volatility adjustment based on time and asset type"""
        if asset_type == 'crypto':
            # 24/7 market, higher volatility during US/Asia overlap
            if 0 <= hour <= 4 or 20 <= hour <= 23:  # Asia/US overlap
                return 1.3
            else:
                return 1.0
        elif asset_type == 'forex':
            # Higher during major market overlaps
            if 8 <= hour <= 12:  # London/NY overlap
                return 1.4
            elif 0 <= hour <= 4:  # Asia session
                return 1.2
            else:
                return 0.8
        elif asset_type in ['stock', 'index']:
            # Market hours volatility
            if 14 <= hour <= 16:  # US market open
                return 1.5
            elif 20 <= hour <= 21:  # US market close
                return 1.3
            elif 9 <= hour <= 17:  # Regular hours
                return 1.0
            else:
                return 0.3  # After hours
        else:  # commodity
            return 1.0
    
    def _calculate_trend(self, ticker: str, timeframe: str) -> float:
        """Calculate trend strength for trending assets"""
        history = self.price_history[ticker][timeframe]
        if len(history) < 5:
            return np.random.choice([-1, 1]) * np.random.uniform(0.3, 0.7)
        
        # Calculate recent trend
        recent_changes = [h['change'] for h in history[-5:]]
        trend_strength = np.mean(recent_changes) * 10
        
        # Add momentum
        if abs(trend_strength) > 0.5:
            trend_strength *= 1.2
        
        # Random trend changes
        if np.random.random() < 0.1:  # 10% chance of trend reversal
            trend_strength *= -1
            
        return np.clip(trend_strength, -1, 1)
    
    def _calculate_mean_reversion(self, ticker: str, current: float, mean: float) -> float:
        """Calculate mean reversion force"""
        deviation = (current - mean) / mean
        reversion_force = -deviation * 2  # Stronger reversion for larger deviations
        
        # Add some randomness
        reversion_force += np.random.uniform(-0.2, 0.2)
        
        return np.clip(reversion_force, -1, 1)
    
    def _calculate_cycle(self, timeframe: str, current_time: datetime) -> float:
        """Calculate cyclical patterns for commodities"""
        # Different cycles for different timeframes
        if timeframe == '15min':
            cycle_period = 96  # One day in 15min candles
        elif timeframe == '1hour':
            cycle_period = 168  # One week in hourly candles
        elif timeframe == '4hour':
            cycle_period = 42  # One week in 4h candles
        else:  # 1day
            cycle_period = 30  # One month
            
        # Simple sine wave with noise
        minutes_since_epoch = int(current_time.timestamp() / 60)
        cycle_position = (minutes_since_epoch % cycle_period) / cycle_period
        cycle_value = np.sin(2 * np.pi * cycle_position)
        
        # Add noise
        cycle_value += np.random.uniform(-0.3, 0.3)
        
        return np.clip(cycle_value, -1, 1)
    
    def _get_market_hours_factor(self, current_time: datetime) -> float:
        """Get market hours volatility factor"""
        hour = current_time.hour
        weekday = current_time.weekday()
        
        # Weekend - very low volatility
        if weekday >= 5:
            return 0.2
            
        # Market hours (simplified for US market)
        if 14 <= hour <= 21:  # 9:30 AM - 4:00 PM EST in UTC
            if 14 <= hour <= 15:  # First hour
                return 1.5
            elif 20 <= hour <= 21:  # Last hour
                return 1.3
            else:
                return 1.0
        else:
            return 0.3  # After hours
    
    def set_volatility_regime(self, ticker: str, regime: str):
        """Set volatility regime for a ticker"""
        regime_targets = {
            'low': 0.5,
            'normal': 1.0,
            'high': 1.5,
            'extreme': 2.0
        }
        self.volatility_state[ticker]['target'] = regime_targets.get(regime, 1.0)

# =============================================================================
# TIMEFRAME DATA MANAGER
# =============================================================================

class TimeframeDataManager:
    """Manages data across multiple timeframes"""
    
    def __init__(self):
        self.data_cache = defaultdict(dict)
        self.cache_timestamps = defaultdict(dict)
        self.price_calculator = TimeframePriceCalculator()
        self.fmp_provider = FMPDataProvider(FMP_API_KEY) if BACKEND_AVAILABLE and FMP_API_KEY else None
        
    def get_timeframe_data(self, ticker: str, timeframe: str, 
                          current_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Get comprehensive data for a specific timeframe
        """
        config = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG['1day'])
        
        # Check cache
        cache_key = f"{ticker}_{timeframe}"
        if self._is_cache_valid(cache_key, config['update_frequency']):
            logger.debug(f"Using cached data for {ticker} {timeframe}")
            return self.data_cache[ticker][timeframe]
        
        # Get or calculate current price for this timeframe
        if current_price is None:
            current_price = self._get_current_price(ticker)
            
        timeframe_price = self.price_calculator.calculate_timeframe_price(
            ticker, current_price, timeframe
        )
        
        # Generate historical data for this timeframe
        historical_data = self._generate_timeframe_history(
            ticker, timeframe_price, timeframe, config
        )
        
        # Calculate technical indicators
        indicators = self._calculate_timeframe_indicators(historical_data)
        
        # Calculate price changes
        price_changes = self._calculate_price_changes(
            historical_data, timeframe, config
        )
        
        # Compile comprehensive data
        timeframe_data = {
            'ticker': ticker,
            'timeframe': timeframe,
            'current_price': timeframe_price,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'historical_data': historical_data,
            'indicators': indicators,
            'price_changes': price_changes,
            'next_update': datetime.now() + timedelta(seconds=config['update_frequency'])
        }
        
        # Cache the data
        self.data_cache[ticker][timeframe] = timeframe_data
        self.cache_timestamps[ticker][timeframe] = datetime.now()
        
        return timeframe_data
    
    def get_multi_timeframe_analysis(self, ticker: str, 
                                   current_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Get analysis across all timeframes
        """
        analysis = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'timeframes': {}
        }
        
        for timeframe in TIMEFRAME_CONFIG.keys():
            analysis['timeframes'][timeframe] = self.get_timeframe_data(
                ticker, timeframe, current_price
            )
        
        # Add cross-timeframe analysis
        analysis['cross_timeframe'] = self._analyze_cross_timeframe(analysis['timeframes'])
        
        return analysis
    
    def _is_cache_valid(self, cache_key: str, max_age_seconds: int) -> bool:
        """Check if cached data is still valid"""
        parts = cache_key.split('_')
        if len(parts) >= 2:
            ticker, timeframe = parts[0], parts[1]
            if ticker in self.cache_timestamps and timeframe in self.cache_timestamps[ticker]:
                age = (datetime.now() - self.cache_timestamps[ticker][timeframe]).seconds
                return age < max_age_seconds
        return False
    
    def _get_current_price(self, ticker: str) -> float:
        """Get current price with fallback"""
        try:
            if self.fmp_provider:
                return self.fmp_provider.fetch_real_time_price(ticker)
        except:
            pass
            
        # Fallback to generated price
        min_price, max_price = self.price_calculator._get_price_bounds(ticker)
        return (min_price + max_price) / 2
    
    def _generate_timeframe_history(self, ticker: str, current_price: float, 
                                  timeframe: str, config: Dict) -> pd.DataFrame:
        """Generate historical data for timeframe"""
        periods = config['candles_per_day'] * config['lookback_days']
        
        # Generate timestamps
        now = datetime.now()
        timestamps = []
        for i in range(periods, 0, -1):
            timestamps.append(now - timedelta(minutes=i * config['period_minutes']))
        
        # Generate prices
        prices = []
        price = current_price
        asset_type = self.price_calculator._get_asset_type(ticker)
        max_change = config['max_change_per_period'].get(asset_type, 0.02)
        
        for i, timestamp in enumerate(timestamps):
            # Generate OHLC data
            volatility = max_change * config['volatility_multiplier'] * 0.5
            
            # Calculate price with some persistence
            if i > 0 and prices:
                last_close = prices[-1]['Close']
                trend = (price - last_close) / last_close
                price = last_close * (1 + trend * 0.7 + np.random.normal(0, volatility))
            
            high = price * (1 + abs(np.random.normal(0, volatility * 0.3)))
            low = price * (1 - abs(np.random.normal(0, volatility * 0.3)))
            open_price = low + (high - low) * np.random.random()
            close = low + (high - low) * np.random.random()
            
            # Volume based on asset type and timeframe
            if asset_type == 'crypto':
                base_volume = np.random.randint(1000000, 50000000)
            elif asset_type == 'forex':
                base_volume = np.random.randint(10000000, 500000000)
            else:
                base_volume = np.random.randint(500000, 20000000)
                
            volume = base_volume * (1 + np.random.uniform(-0.5, 1.0))
            
            prices.append({
                'timestamp': timestamp,
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': int(volume)
            })
            
            price = close
        
        df = pd.DataFrame(prices)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def _calculate_timeframe_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for timeframe"""
        if df.empty or len(df) < 20:
            return {}
            
        indicators = {}
        
        try:
            # Price-based indicators
            indicators['sma_20'] = df['Close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = df['Close'].rolling(min(50, len(df))).mean().iloc[-1]
            indicators['ema_20'] = df['Close'].ewm(span=20).mean().iloc[-1]
            
            # RSI
            if BACKEND_AVAILABLE:
                indicators['rsi'] = calculate_rsi(df['Close']).iloc[-1]
            else:
                # Simple RSI calculation
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
            
            # MACD
            if BACKEND_AVAILABLE:
                macd, signal = calculate_macd(df['Close'])
                indicators['macd'] = macd.iloc[-1]
                indicators['macd_signal'] = signal.iloc[-1]
            else:
                # Simple MACD
                exp1 = df['Close'].ewm(span=12).mean()
                exp2 = df['Close'].ewm(span=26).mean()
                macd = exp1 - exp2
                indicators['macd'] = macd.iloc[-1]
                indicators['macd_signal'] = macd.ewm(span=9).mean().iloc[-1]
            
            # Bollinger Bands
            if BACKEND_AVAILABLE:
                upper, middle, lower = calculate_bollinger_bands(df['Close'])
                indicators['bb_upper'] = upper.iloc[-1]
                indicators['bb_middle'] = middle.iloc[-1]
                indicators['bb_lower'] = lower.iloc[-1]
            else:
                # Simple BB
                sma = df['Close'].rolling(20).mean()
                std = df['Close'].rolling(20).std()
                indicators['bb_upper'] = (sma + 2 * std).iloc[-1]
                indicators['bb_middle'] = sma.iloc[-1]
                indicators['bb_lower'] = (sma - 2 * std).iloc[-1]
            
            # Volume indicators
            indicators['volume_sma'] = df['Volume'].rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = df['Volume'].iloc[-1] / indicators['volume_sma']
            
            # Volatility
            returns = df['Close'].pct_change()
            indicators['volatility'] = returns.std() * np.sqrt(252)  # Annualized
            
            # Support/Resistance (simplified)
            indicators['resistance'] = df['High'].rolling(50).max().iloc[-1]
            indicators['support'] = df['Low'].rolling(50).min().iloc[-1]
            
        except Exception as e:
            logger.warning(f"Error calculating indicators: {e}")
            
        return indicators
    
    def _calculate_price_changes(self, df: pd.DataFrame, timeframe: str, 
                               config: Dict) -> Dict[str, float]:
        """Calculate price changes for different periods"""
        if df.empty:
            return {}
            
        changes = {}
        current_price = df['Close'].iloc[-1]
        
        try:
            # Period change
            if len(df) > 1:
                period_change = (current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
                changes['period'] = period_change * 100
            
            # Hourly change (for sub-daily timeframes)
            if config['period_minutes'] < 60:
                periods_per_hour = 60 // config['period_minutes']
                if len(df) > periods_per_hour:
                    hour_ago_price = df['Close'].iloc[-periods_per_hour-1]
                    changes['hourly'] = ((current_price - hour_ago_price) / hour_ago_price) * 100
            
            # Daily change
            if config['period_minutes'] < 1440:
                periods_per_day = config['candles_per_day']
                if len(df) > periods_per_day:
                    day_ago_price = df['Close'].iloc[-periods_per_day-1]
                    changes['daily'] = ((current_price - day_ago_price) / day_ago_price) * 100
            
            # Weekly change
            periods_per_week = config['candles_per_day'] * 7
            if len(df) > periods_per_week:
                week_ago_price = df['Close'].iloc[-periods_per_week-1]
                changes['weekly'] = ((current_price - week_ago_price) / week_ago_price) * 100
            
            # High/Low ratios
            if config['period_minutes'] <= 1440:  # For intraday timeframes
                daily_high = df['High'].tail(config['candles_per_day']).max()
                daily_low = df['Low'].tail(config['candles_per_day']).min()
                changes['daily_range'] = ((daily_high - daily_low) / daily_low) * 100
                changes['position_in_range'] = ((current_price - daily_low) / (daily_high - daily_low)) * 100
            
        except Exception as e:
            logger.warning(f"Error calculating price changes: {e}")
            
        return changes
    
    def _analyze_cross_timeframe(self, timeframes_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze patterns across timeframes"""
        analysis = {
            'trend_alignment': self._check_trend_alignment(timeframes_data),
            'volatility_comparison': self._compare_volatility(timeframes_data),
            'support_resistance_levels': self._identify_key_levels(timeframes_data),
            'momentum_divergence': self._check_momentum_divergence(timeframes_data),
            'timeframe_strength': self._calculate_timeframe_strength(timeframes_data)
        }
        
        return analysis
    
    def _check_trend_alignment(self, timeframes_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Check if trends align across timeframes"""
        trends = {}
        
        for tf, data in timeframes_data.items():
            if 'indicators' in data and data['indicators']:
                indicators = data['indicators']
                current_price = data['current_price']
                
                # Simple trend determination
                if 'sma_20' in indicators and 'sma_50' in indicators:
                    if current_price > indicators['sma_20'] > indicators['sma_50']:
                        trends[tf] = 'bullish'
                    elif current_price < indicators['sma_20'] < indicators['sma_50']:
                        trends[tf] = 'bearish'
                    else:
                        trends[tf] = 'neutral'
                        
        # Check alignment
        trend_values = list(trends.values())
        if len(trend_values) > 0:
            bullish_count = trend_values.count('bullish')
            bearish_count = trend_values.count('bearish')
            
            if bullish_count > len(trend_values) * 0.7:
                alignment = 'strong_bullish'
            elif bearish_count > len(trend_values) * 0.7:
                alignment = 'strong_bearish'
            elif bullish_count > bearish_count:
                alignment = 'weak_bullish'
            elif bearish_count > bullish_count:
                alignment = 'weak_bearish'
            else:
                alignment = 'mixed'
        else:
            alignment = 'unknown'
            
        return {
            'trends': trends,
            'alignment': alignment,
            'strength': max(bullish_count, bearish_count) / len(trend_values) if trend_values else 0
        }
    
    def _compare_volatility(self, timeframes_data: Dict[str, Dict]) -> Dict[str, float]:
        """Compare volatility across timeframes"""
        volatilities = {}
        
        for tf, data in timeframes_data.items():
            if 'indicators' in data and 'volatility' in data['indicators']:
                volatilities[tf] = data['indicators']['volatility']
                
        return volatilities
    
    def _identify_key_levels(self, timeframes_data: Dict[str, Dict]) -> Dict[str, List[float]]:
        """Identify key support/resistance levels"""
        all_supports = []
        all_resistances = []
        
        for tf, data in timeframes_data.items():
            if 'indicators' in data:
                if 'support' in data['indicators']:
                    all_supports.append(data['indicators']['support'])
                if 'resistance' in data['indicators']:
                    all_resistances.append(data['indicators']['resistance'])
                    
        # Cluster nearby levels
        key_supports = self._cluster_levels(all_supports)
        key_resistances = self._cluster_levels(all_resistances)
        
        return {
            'supports': sorted(key_supports),
            'resistances': sorted(key_resistances)
        }
    
    def _cluster_levels(self, levels: List[float], threshold: float = 0.02) -> List[float]:
        """Cluster nearby price levels"""
        if not levels:
            return []
            
        sorted_levels = sorted(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
                
        clusters.append(np.mean(current_cluster))
        return clusters
    
    def _check_momentum_divergence(self, timeframes_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Check for momentum divergence across timeframes"""
        momentum_indicators = {}
        
        for tf, data in timeframes_data.items():
            if 'indicators' in data:
                indicators = data['indicators']
                momentum = {}
                
                if 'rsi' in indicators:
                    momentum['rsi'] = indicators['rsi']
                if 'macd' in indicators:
                    momentum['macd'] = indicators['macd']
                    
                if momentum:
                    momentum_indicators[tf] = momentum
                    
        # Check for divergences
        divergences = []
        timeframe_list = list(momentum_indicators.keys())
        
        for i in range(len(timeframe_list) - 1):
            tf1 = timeframe_list[i]
            tf2 = timeframe_list[i + 1]
            
            if 'rsi' in momentum_indicators[tf1] and 'rsi' in momentum_indicators[tf2]:
                rsi_diff = momentum_indicators[tf1]['rsi'] - momentum_indicators[tf2]['rsi']
                if abs(rsi_diff) > 20:
                    divergences.append({
                        'type': 'rsi',
                        'timeframes': [tf1, tf2],
                        'difference': rsi_diff
                    })
                    
        return {
            'momentum_by_timeframe': momentum_indicators,
            'divergences': divergences,
            'has_divergence': len(divergences) > 0
        }
    
    def _calculate_timeframe_strength(self, timeframes_data: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate signal strength for each timeframe"""
        strengths = {}
        
        for tf, data in timeframes_data.items():
            strength = 0
            count = 0
            
            if 'indicators' in data:
                indicators = data['indicators']
                
                # RSI strength
                if 'rsi' in indicators:
                    if indicators['rsi'] > 70:
                        strength += 1  # Overbought
                    elif indicators['rsi'] < 30:
                        strength -= 1  # Oversold
                    count += 1
                    
                # MACD strength
                if 'macd' in indicators and 'macd_signal' in indicators:
                    if indicators['macd'] > indicators['macd_signal']:
                        strength += 0.5
                    else:
                        strength -= 0.5
                    count += 1
                    
                # Moving average strength
                if 'sma_20' in indicators and 'current_price' in data:
                    if data['current_price'] > indicators['sma_20']:
                        strength += 0.5
                    else:
                        strength -= 0.5
                    count += 1
                    
            if count > 0:
                strengths[tf] = strength / count
            else:
                strengths[tf] = 0
                
        return strengths

# =============================================================================
# REAL-TIME PRICE UPDATER
# =============================================================================

class RealTimePriceUpdater:
    """Handles real-time price updates for different timeframes"""
    
    def __init__(self, timeframe_manager: TimeframeDataManager):
        self.timeframe_manager = timeframe_manager
        self.update_callbacks = defaultdict(list)
        self.last_update_times = defaultdict(dict)
        self.is_running = False
        
    def start_updates(self, tickers: List[str], timeframes: List[str]):
        """Start real-time updates for specified tickers and timeframes"""
        self.is_running = True
        
        for ticker in tickers:
            for timeframe in timeframes:
                self._schedule_update(ticker, timeframe)
                
    def stop_updates(self):
        """Stop all real-time updates"""
        self.is_running = False
        
    def register_callback(self, ticker: str, timeframe: str, callback):
        """Register callback for price updates"""
        key = f"{ticker}_{timeframe}"
        self.update_callbacks[key].append(callback)
        
    def _schedule_update(self, ticker: str, timeframe: str):
        """Schedule periodic updates based on timeframe"""
        config = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG['1day'])
        update_frequency = config['update_frequency']
        
        def update_task():
            while self.is_running:
                try:
                    # Get updated data
                    data = self.timeframe_manager.get_timeframe_data(ticker, timeframe)
                    
                    # Trigger callbacks
                    key = f"{ticker}_{timeframe}"
                    for callback in self.update_callbacks[key]:
                        callback(data)
                        
                    # Update timestamp
                    self.last_update_times[ticker][timeframe] = datetime.now()
                    
                except Exception as e:
                    logger.error(f"Error updating {ticker} {timeframe}: {e}")
                    
                time.sleep(update_frequency)
                
        # Start update thread (in production, use proper async)
        import threading
        thread = threading.Thread(target=update_task, daemon=True)
        thread.start()

# =============================================================================
# INTEGRATION FUNCTIONS FOR STREAMLIT
# =============================================================================

def get_timeframe_prediction(ticker: str, timeframe: str, models=None, config=None) -> Dict[str, Any]:
    """
    Get prediction adjusted for specific timeframe
    Integrates with fixedui.py prediction system
    """
    try:
        manager = TimeframeDataManager()
        
        # Get timeframe-specific data
        tf_data = manager.get_timeframe_data(ticker, timeframe)
        
        if not tf_data:
            logger.error(f"No data available for {ticker} {timeframe}")
            return None
            
        current_price = tf_data['current_price']
        indicators = tf_data.get('indicators', {})
        price_changes = tf_data.get('price_changes', {})
        
        # Prepare prediction result compatible with fixedui.py
        prediction_result = {
            'ticker': ticker,
            'timeframe': timeframe,
            'asset_type': manager.price_calculator._get_asset_type(ticker),
            'current_price': current_price,
            'timestamp': datetime.now().isoformat(),
            
            # Timeframe-specific metrics
            'timeframe_data': {
                'period_change': price_changes.get('period', 0),
                'daily_change': price_changes.get('daily', 0),
                'volatility': indicators.get('volatility', 0),
                'volume_ratio': indicators.get('volume_ratio', 1.0)
            },
            
            # Technical indicators
            'technical_indicators': {
                'rsi': indicators.get('rsi', 50),
                'macd': indicators.get('macd', 0),
                'macd_signal': indicators.get('macd_signal', 0),
                'bb_upper': indicators.get('bb_upper', current_price * 1.02),
                'bb_lower': indicators.get('bb_lower', current_price * 0.98),
                'sma_20': indicators.get('sma_20', current_price),
                'sma_50': indicators.get('sma_50', current_price)
            },
            
            # Support/Resistance
            'key_levels': {
                'support': indicators.get('support', current_price * 0.95),
                'resistance': indicators.get('resistance', current_price * 1.05)
            }
        }
        
        # If models provided, integrate with backend prediction
        if models and BACKEND_AVAILABLE:
            try:
                from enhprog import get_real_time_prediction
                
                # Get AI prediction
                ai_prediction = get_real_time_prediction(ticker, models, config, current_price)
                
                if ai_prediction:
                    # Merge AI prediction with timeframe data
                    prediction_result.update({
                        'predicted_price': ai_prediction.get('predicted_price', current_price),
                        'confidence': ai_prediction.get('confidence', 50),
                        'price_change': ai_prediction.get('price_change', 0),
                        'price_change_pct': ai_prediction.get('price_change_pct', 0),
                        'forecast_5_day': ai_prediction.get('forecast_5_day', []),
                        'models_used': ai_prediction.get('models_used', []),
                        'risk_metrics': ai_prediction.get('risk_metrics', {})
                    })
            except Exception as e:
                logger.warning(f"Could not get AI prediction: {e}")
                
        # Add timeframe-specific prediction adjustments
        if 'predicted_price' not in prediction_result:
            # Simple prediction based on timeframe momentum
            momentum = 0
            if indicators.get('rsi', 50) > 60:
                momentum += 0.3
            elif indicators.get('rsi', 50) < 40:
                momentum -= 0.3
                
            if indicators.get('macd', 0) > indicators.get('macd_signal', 0):
                momentum += 0.2
            else:
                momentum -= 0.2
                
            tf_config = TIMEFRAME_CONFIG[timeframe]
            asset_type = prediction_result['asset_type']
            max_change = tf_config['max_change_per_period'].get(asset_type, 0.02)
            
            predicted_change = momentum * max_change * 0.5
            prediction_result['predicted_price'] = current_price * (1 + predicted_change)
            prediction_result['price_change'] = prediction_result['predicted_price'] - current_price
            prediction_result['price_change_pct'] = predicted_change * 100
            prediction_result['confidence'] = 50 + abs(momentum) * 30
            
        return prediction_result
        
    except Exception as e:
        logger.error(f"Error getting timeframe prediction: {e}")
        return None

def get_multi_timeframe_view(ticker: str) -> Dict[str, Any]:
    """
    Get synchronized view across all timeframes
    For use in Streamlit multi-timeframe display
    """
    try:
        manager = TimeframeDataManager()
        
        # Get analysis across all timeframes
        analysis = manager.get_multi_timeframe_analysis(ticker)
        
        # Prepare view data
        view_data = {
            'ticker': ticker,
            'timestamp': analysis['timestamp'],
            'timeframes': {},
            'summary': {}
        }
        
        # Process each timeframe
        for tf, tf_data in analysis['timeframes'].items():
            view_data['timeframes'][tf] = {
                'current_price': tf_data['current_price'],
                'period_change': tf_data['price_changes'].get('period', 0),
                'daily_change': tf_data['price_changes'].get('daily', 0),
                'indicators': tf_data.get('indicators', {}),
                'config': tf_data['config']
            }
            
        # Add cross-timeframe analysis
        if 'cross_timeframe' in analysis:
            cross = analysis['cross_timeframe']
            view_data['summary'] = {
                'trend_alignment': cross.get('trend_alignment', {}).get('alignment', 'unknown'),
                'trend_strength': cross.get('trend_alignment', {}).get('strength', 0),
                'key_supports': cross.get('support_resistance_levels', {}).get('supports', []),
                'key_resistances': cross.get('support_resistance_levels', {}).get('resistances', []),
                'has_divergence': cross.get('momentum_divergence', {}).get('has_divergence', False)
            }
            
        return view_data
        
    except Exception as e:
        logger.error(f"Error getting multi-timeframe view: {e}")
        return None

def update_streamlit_prices(ticker: str, timeframe: str, price_callback):
    """
    Real-time price updates for Streamlit
    Calls the callback with updated prices based on timeframe
    """
    updater = RealTimePriceUpdater(TimeframeDataManager())
    updater.register_callback(ticker, timeframe, price_callback)
    updater.start_updates([ticker], [timeframe])
    
    return updater

# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    logger.info("Testing Timeframe Management System")
    
    # Test tickers
    test_tickers = ["BTCUSD", "^GSPC", "USDJPY", "GC=F"]
    
    # Initialize manager
    manager = TimeframeDataManager()
    
    for ticker in test_tickers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {ticker}")
        logger.info(f"{'='*60}")
        
        # Test each timeframe
        for timeframe in ['15min', '1hour', '4hour', '1day']:
            logger.info(f"\nTimeframe: {timeframe}")
            
            # Get timeframe data
            data = manager.get_timeframe_data(ticker, timeframe)
            
            if data:
                logger.info(f"Current Price: ${data['current_price']:.4f}")
                logger.info(f"Period Change: {data['price_changes'].get('period', 0):.2f}%")
                
                indicators = data.get('indicators', {})
                if indicators:
                    logger.info(f"RSI: {indicators.get('rsi', 'N/A'):.2f}")
                    logger.info(f"Volatility: {indicators.get('volatility', 'N/A'):.2%}")
                    
        # Test multi-timeframe analysis
        logger.info("\nMulti-Timeframe Analysis:")
        analysis = manager.get_multi_timeframe_analysis(ticker)
        
        if analysis and 'cross_timeframe' in analysis:
            cross = analysis['cross_timeframe']
            logger.info(f"Trend Alignment: {cross['trend_alignment']['alignment']}")
            logger.info(f"Key Supports: {cross['support_resistance_levels']['supports']}")
            logger.info(f"Key Resistances: {cross['support_resistance_levels']['resistances']}")
            
    logger.info("\n✅ Timeframe Management System Test Complete")