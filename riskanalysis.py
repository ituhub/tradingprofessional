"""
RISK ANALYSIS MODULE - COMPREHENSIVE RISK MANAGEMENT SYSTEM
==============================================================================
Advanced risk analysis with multiple risk metrics, portfolio risk management,
and real-time risk monitoring capabilities.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Data class for risk metrics"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float
    downside_deviation: float
    upside_potential: float
    omega_ratio: float
    risk_adjusted_return: float

class AdvancedRiskAnalyzer:
    """Advanced Risk Analysis Engine"""
    
    def __init__(self):
        """Initialize the risk analyzer"""
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.trading_days = 252
        self.confidence_levels = [0.95, 0.99]
        
        # Market benchmark data (simulated S&P 500 returns)
        self.benchmark_returns = self._generate_benchmark_returns()
        
        logger.info("AdvancedRiskAnalyzer initialized successfully")
    
    def _generate_benchmark_returns(self, days: int = 252) -> np.ndarray:
        """Generate simulated benchmark returns for comparison"""
        np.random.seed(42)  # For consistency
        # Simulate S&P 500-like returns: ~10% annual return, ~16% volatility
        daily_return = 0.10 / 252
        daily_vol = 0.16 / np.sqrt(252)
        
        returns = np.random.normal(daily_return, daily_vol, days)
        return returns
    
    def calculate_comprehensive_risk_metrics(
        self, 
        price_data: np.ndarray, 
        prediction: Dict[str, Any],
        portfolio_value: float = 100000
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics
        
        Args:
            price_data: Historical price data
            prediction: Current prediction dictionary
            portfolio_value: Portfolio value for position sizing
            
        Returns:
            RiskMetrics: Comprehensive risk metrics
        """
        try:
            # Calculate returns
            returns = self._calculate_returns(price_data)
            
            if len(returns) < 30:
                logger.warning("Insufficient data for comprehensive risk analysis")
                return self._generate_fallback_metrics()
            
            # Basic risk metrics
            volatility = self._calculate_volatility(returns)
            var_95, var_99 = self._calculate_var(returns)
            cvar_95, cvar_99 = self._calculate_conditional_var(returns)
            max_drawdown = self._calculate_max_drawdown(price_data)
            
            # Performance ratios
            sharpe_ratio = self._calculate_sharpe_ratio(returns, volatility)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
            
            # Market risk metrics
            beta, alpha = self._calculate_beta_alpha(returns)
            information_ratio = self._calculate_information_ratio(returns)
            tracking_error = self._calculate_tracking_error(returns)
            
            # Additional risk metrics
            downside_deviation = self._calculate_downside_deviation(returns)
            upside_potential = self._calculate_upside_potential(returns)
            omega_ratio = self._calculate_omega_ratio(returns)
            risk_adjusted_return = self._calculate_risk_adjusted_return(returns, volatility)
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                beta=beta,
                alpha=alpha,
                information_ratio=information_ratio,
                tracking_error=tracking_error,
                downside_deviation=downside_deviation,
                upside_potential=upside_potential,
                omega_ratio=omega_ratio,
                risk_adjusted_return=risk_adjusted_return
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return self._generate_fallback_metrics()
    
    def _calculate_returns(self, price_data: np.ndarray) -> np.ndarray:
        """Calculate percentage returns from price data"""
        if len(price_data) < 2:
            return np.array([])
        
        returns = np.diff(price_data) / price_data[:-1]
        return returns[~np.isnan(returns)]  # Remove NaN values
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility"""
        if len(returns) == 0:
            return 0.0
        return np.std(returns) * np.sqrt(self.trading_days)
    
    def _calculate_var(self, returns: np.ndarray, confidence_levels: List[float] = None) -> Tuple[float, float]:
        """Calculate Value at Risk at different confidence levels"""
        if len(returns) == 0:
            return 0.0, 0.0
        
        if confidence_levels is None:
            confidence_levels = self.confidence_levels
        
        var_95 = np.percentile(returns, (1 - confidence_levels[0]) * 100)
        var_99 = np.percentile(returns, (1 - confidence_levels[1]) * 100)
        
        return var_95, var_99
    
    def _calculate_conditional_var(self, returns: np.ndarray) -> Tuple[float, float]:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0, 0.0
        
        var_95, var_99 = self._calculate_var(returns)
        
        # CVaR is the expected value of returns below VaR threshold
        cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95
        cvar_99 = np.mean(returns[returns <= var_99]) if np.any(returns <= var_99) else var_99
        
        return cvar_95, cvar_99
    
    def _calculate_max_drawdown(self, price_data: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(price_data) < 2:
            return 0.0
        
        # Calculate cumulative returns
        cumulative = np.cumprod(1 + self._calculate_returns(price_data))
        running_max = np.maximum.accumulate(cumulative)
        
        # Calculate drawdowns
        drawdowns = (cumulative - running_max) / running_max
        
        return np.min(drawdowns) if len(drawdowns) > 0 else 0.0
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, volatility: float) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or volatility == 0:
            return 0.0
        
        excess_return = np.mean(returns) * self.trading_days - self.risk_free_rate
        return excess_return / volatility
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (focuses on downside deviation)"""
        if len(returns) == 0:
            return 0.0
        
        excess_return = np.mean(returns) * self.trading_days - self.risk_free_rate
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if excess_return > 0 else 0.0
        
        downside_deviation = np.std(downside_returns) * np.sqrt(self.trading_days)
        
        return excess_return / downside_deviation if downside_deviation != 0 else 0.0
    
    def _calculate_calmar_ratio(self, returns: np.ndarray, max_drawdown: float) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        if len(returns) == 0 or max_drawdown == 0:
            return 0.0
        
        annual_return = np.mean(returns) * self.trading_days
        return annual_return / abs(max_drawdown)
    
    def _calculate_beta_alpha(self, returns: np.ndarray) -> Tuple[float, float]:
        """Calculate beta and alpha relative to benchmark"""
        if len(returns) == 0:
            return 1.0, 0.0
        
        # Align returns with benchmark
        min_length = min(len(returns), len(self.benchmark_returns))
        asset_returns = returns[-min_length:]
        benchmark_returns = self.benchmark_returns[-min_length:]
        
        if min_length < 10:
            return 1.0, 0.0
        
        # Calculate beta using linear regression
        covariance = np.cov(asset_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 1.0
        
        # Calculate alpha
        asset_mean = np.mean(asset_returns) * self.trading_days
        benchmark_mean = np.mean(benchmark_returns) * self.trading_days
        alpha = asset_mean - (self.risk_free_rate + beta * (benchmark_mean - self.risk_free_rate))
        
        return beta, alpha
    
    def _calculate_information_ratio(self, returns: np.ndarray) -> float:
        """Calculate information ratio"""
        if len(returns) == 0:
            return 0.0
        
        min_length = min(len(returns), len(self.benchmark_returns))
        asset_returns = returns[-min_length:]
        benchmark_returns = self.benchmark_returns[-min_length:]
        
        if min_length < 10:
            return 0.0
        
        excess_returns = asset_returns - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(self.trading_days)
        
        if tracking_error == 0:
            return 0.0
        
        return (np.mean(excess_returns) * self.trading_days) / tracking_error
    
    def _calculate_tracking_error(self, returns: np.ndarray) -> float:
        """Calculate tracking error relative to benchmark"""
        if len(returns) == 0:
            return 0.0
        
        min_length = min(len(returns), len(self.benchmark_returns))
        asset_returns = returns[-min_length:]
        benchmark_returns = self.benchmark_returns[-min_length:]
        
        if min_length < 10:
            return 0.0
        
        excess_returns = asset_returns - benchmark_returns
        return np.std(excess_returns) * np.sqrt(self.trading_days)
    
    def _calculate_downside_deviation(self, returns: np.ndarray) -> float:
        """Calculate downside deviation"""
        if len(returns) == 0:
            return 0.0
        
        target_return = self.risk_free_rate / self.trading_days
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return 0.0
        
        return np.std(downside_returns) * np.sqrt(self.trading_days)
    
    def _calculate_upside_potential(self, returns: np.ndarray) -> float:
        """Calculate upside potential"""
        if len(returns) == 0:
            return 0.0
        
        target_return = self.risk_free_rate / self.trading_days
        upside_returns = returns[returns > target_return]
        
        if len(upside_returns) == 0:
            return 0.0
        
        return np.mean(upside_returns) * self.trading_days
    
    def _calculate_omega_ratio(self, returns: np.ndarray, threshold: float = None) -> float:
        """Calculate Omega ratio"""
        if len(returns) == 0:
            return 1.0
        
        if threshold is None:
            threshold = self.risk_free_rate / self.trading_days
        
        gains = returns[returns > threshold]
        losses = returns[returns <= threshold]
        
        if len(losses) == 0:
            return float('inf')
        
        if len(gains) == 0:
            return 0.0
        
        probability_weighted_gains = np.sum(gains - threshold)
        probability_weighted_losses = np.sum(threshold - losses)
        
        return probability_weighted_gains / probability_weighted_losses if probability_weighted_losses != 0 else 0.0
    
    def _calculate_risk_adjusted_return(self, returns: np.ndarray, volatility: float) -> float:
        """Calculate risk-adjusted return"""
        if len(returns) == 0 or volatility == 0:
            return 0.0
        
        annual_return = np.mean(returns) * self.trading_days
        return annual_return / volatility
    
    def _generate_fallback_metrics(self) -> RiskMetrics:
        """Generate fallback metrics when calculation fails"""
        return RiskMetrics(
            var_95=-0.02,
            var_99=-0.035,
            cvar_95=-0.025,
            cvar_99=-0.04,
            volatility=0.20,
            sharpe_ratio=0.8,
            sortino_ratio=1.1,
            calmar_ratio=1.5,
            max_drawdown=-0.15,
            beta=1.0,
            alpha=0.02,
            information_ratio=0.5,
            tracking_error=0.05,
            downside_deviation=0.12,
            upside_potential=0.15,
            omega_ratio=1.3,
            risk_adjusted_return=0.5
        )
    
    def calculate_position_sizing(
        self, 
        risk_metrics: RiskMetrics, 
        account_balance: float, 
        risk_per_trade: float,
        current_price: float,
        stop_loss_pct: float
    ) -> Dict[str, Any]:
        """
        Calculate optimal position sizing based on risk metrics
        
        Args:
            risk_metrics: Calculated risk metrics
            account_balance: Account balance
            risk_per_trade: Risk percentage per trade (0.01 = 1%)
            current_price: Current asset price
            stop_loss_pct: Stop loss percentage
            
        Returns:
            Dict with position sizing recommendations
        """
        try:
            # Kelly Criterion calculation
            win_rate = 0.55  # Assume 55% win rate
            avg_win = abs(risk_metrics.upside_potential)
            avg_loss = abs(risk_metrics.var_95)
            
            if avg_loss != 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            else:
                kelly_fraction = 0.02
            
            # Volatility-based position sizing
            vol_based_size = min(0.1, 0.02 / risk_metrics.volatility) if risk_metrics.volatility > 0 else 0.02
            
            # VaR-based position sizing
            var_based_size = min(0.15, risk_per_trade / abs(risk_metrics.var_95)) if risk_metrics.var_95 < 0 else risk_per_trade
            
            # Final position size (conservative approach)
            recommended_size = min(kelly_fraction, vol_based_size, var_based_size, risk_per_trade * 2)
            
            # Calculate actual position
            risk_amount = account_balance * recommended_size
            stop_loss_amount = current_price * stop_loss_pct
            
            if stop_loss_amount > 0:
                shares = int(risk_amount / stop_loss_amount)
                position_value = shares * current_price
            else:
                shares = 0
                position_value = 0
            
            return {
                'recommended_size_pct': recommended_size,
                'kelly_fraction': kelly_fraction,
                'volatility_based_size': vol_based_size,
                'var_based_size': var_based_size,
                'recommended_shares': shares,
                'position_value': position_value,
                'risk_amount': risk_amount,
                'portfolio_pct': (position_value / account_balance) * 100 if account_balance > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating position sizing: {e}")
            return {
                'recommended_size_pct': 0.02,
                'kelly_fraction': 0.02,
                'volatility_based_size': 0.02,
                'var_based_size': 0.02,
                'recommended_shares': 0,
                'position_value': 0,
                'risk_amount': account_balance * 0.02,
                'portfolio_pct': 0
            }
    
    def generate_risk_report(self, risk_metrics: RiskMetrics, ticker: str) -> str:
        """Generate comprehensive risk analysis report"""
        try:
            report = f"""
COMPREHENSIVE RISK ANALYSIS REPORT
Asset: {ticker}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== MARKET RISK METRICS ===
Value at Risk (95%): {risk_metrics.var_95:.2%}
Value at Risk (99%): {risk_metrics.var_99:.2%}
Conditional VaR (95%): {risk_metrics.cvar_95:.2%}
Conditional VaR (99%): {risk_metrics.cvar_99:.2%}
Volatility (Annualized): {risk_metrics.volatility:.2%}
Maximum Drawdown: {risk_metrics.max_drawdown:.2%}

=== PERFORMANCE RATIOS ===
Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}
Sortino Ratio: {risk_metrics.sortino_ratio:.2f}
Calmar Ratio: {risk_metrics.calmar_ratio:.2f}
Omega Ratio: {risk_metrics.omega_ratio:.2f}
Information Ratio: {risk_metrics.information_ratio:.2f}

=== MARKET EXPOSURE ===
Beta (Market Sensitivity): {risk_metrics.beta:.2f}
Alpha (Excess Return): {risk_metrics.alpha:.2%}
Tracking Error: {risk_metrics.tracking_error:.2%}

=== RISK-RETURN PROFILE ===
Downside Deviation: {risk_metrics.downside_deviation:.2%}
Upside Potential: {risk_metrics.upside_potential:.2%}
Risk-Adjusted Return: {risk_metrics.risk_adjusted_return:.2f}

=== RISK ASSESSMENT ===
Overall Risk Level: {"HIGH" if risk_metrics.volatility > 0.3 else "MEDIUM" if risk_metrics.volatility > 0.2 else "LOW"}
Recommended Action: {"REDUCE EXPOSURE" if risk_metrics.var_95 < -0.05 else "MONITOR CLOSELY" if risk_metrics.var_95 < -0.03 else "ACCEPTABLE RISK"}

=== KEY INSIGHTS ===
• VaR indicates potential daily loss of {abs(risk_metrics.var_95):.2%} with 95% confidence
• Sharpe ratio of {risk_metrics.sharpe_ratio:.2f} {'exceeds' if risk_metrics.sharpe_ratio > 1 else 'falls below'} market benchmark
• Beta of {risk_metrics.beta:.2f} suggests {'higher' if risk_metrics.beta > 1 else 'lower'} volatility than market
• Maximum drawdown of {abs(risk_metrics.max_drawdown):.2%} indicates {'high' if abs(risk_metrics.max_drawdown) > 0.2 else 'moderate'} downside risk
"""
            return report
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return "Error generating risk analysis report."


class RiskVisualization:
    """Risk analysis visualization components"""
    
    @staticmethod
    def create_risk_dashboard(risk_metrics: RiskMetrics, ticker: str) -> go.Figure:
        """Create comprehensive risk dashboard"""
        try:
            # Create subplots with more vertical space
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[
                    'Value at Risk Analysis',
                    'Performance Ratios',
                    'Risk-Return Profile',
                    'Downside Risk Metrics',
                    'Market Exposure',
                    'Risk Decomposition'
                ],
                specs=[
                    [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "indicator"}, {"type": "pie"}]
                ],
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            # VaR Analysis (Row 1, Col 1)
            var_data = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%']
            var_values = [
                abs(risk_metrics.var_95) * 100,
                abs(risk_metrics.var_99) * 100,
                abs(risk_metrics.cvar_95) * 100,
                abs(risk_metrics.cvar_99) * 100
            ]
            
            fig.add_trace(
                go.Bar(
                    x=var_data,
                    y=var_values,
                    name='VaR Metrics',
                    marker_color=['red', 'darkred', 'orange', 'darkorange'],
                    text=[f'{v:.1f}%' for v in var_values],
                    textposition='outside',
                    textfont=dict(size=12),
                    width=0.6
                ),
                row=1, col=1
            )
            
            # Performance Ratios (Row 1, Col 2)
            ratio_data = ['Sharpe', 'Sortino', 'Calmar', 'Omega']
            ratio_values = [
                risk_metrics.sharpe_ratio,
                risk_metrics.sortino_ratio,
                risk_metrics.calmar_ratio,
                risk_metrics.omega_ratio
            ]
            
            fig.add_trace(
                go.Bar(
                    x=ratio_data,
                    y=ratio_values,
                    name='Performance Ratios',
                    marker_color=['blue', 'lightblue', 'green', 'lightgreen'],
                    text=[f'{v:.2f}' for v in ratio_values],
                    textposition='outside',
                    textfont=dict(size=12),
                    width=0.6
                ),
                row=1, col=2
            )
            
            # Risk-Return Scatter (Row 1, Col 3)
            fig.add_trace(
                go.Scatter(
                    x=[risk_metrics.volatility * 100],
                    y=[risk_metrics.upside_potential * 100],
                    mode='markers+text',
                    name=ticker,
                    marker=dict(size=20, color='purple'),
                    text=[f'{ticker}<br>Vol: {risk_metrics.volatility:.1%}<br>Return: {risk_metrics.upside_potential:.1%}'],
                    textposition='top center',
                    textfont=dict(size=12)
                ),
                row=1, col=3
            )
            
            # Downside Risk Metrics (Row 2, Col 1)
            downside_data = ['Max Drawdown', 'Downside Dev', 'Tracking Error']
            downside_values = [
                abs(risk_metrics.max_drawdown) * 100,
                risk_metrics.downside_deviation * 100,
                risk_metrics.tracking_error * 100
            ]
            
            fig.add_trace(
                go.Bar(
                    x=downside_data,
                    y=downside_values,
                    name='Downside Risk',
                    marker_color=['red', 'orange', 'yellow'],
                    text=[f'{v:.1f}%' for v in downside_values],
                    textposition='outside',
                    textfont=dict(size=12),
                    width=0.6
                ),
                row=2, col=1
            )
            
            # Beta Gauge (Row 2, Col 2)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_metrics.beta,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Beta (Market Risk)", 'font': {'size': 14}},
                    delta={'reference': 1.0, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [0, 2], 'tickwidth': 1, 'tickfont': {'size': 12}},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.7], 'color': "lightgreen"},
                            {'range': [0.7, 1.3], 'color': "yellow"},
                            {'range': [1.3, 2], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 1.5
                        }
                    },
                    number={'font': {'size': 20}}
                ),
                row=2, col=2
            )
            
            # Risk Decomposition Pie Chart (Row 2, Col 3)
            risk_components = ['Market Risk', 'Specific Risk', 'Liquidity Risk']
            risk_values = [
                abs(risk_metrics.beta) * 40,
                risk_metrics.tracking_error * 500,
                risk_metrics.volatility * 25
            ]
            
            fig.add_trace(
                go.Pie(
                    labels=risk_components,
                    values=risk_values,
                    name="Risk Decomposition",
                    textinfo='label+percent',
                    textposition='inside',
                    textfont=dict(size=12),
                    hole=0.3
                ),
                row=2, col=3
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                title={
                    'text': f"Comprehensive Risk Analysis Dashboard - {ticker}",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 16}
                },
                showlegend=False,
                margin=dict(t=100, b=50, l=50, r=50)
            )
            
            # Update axes for better readability
            fig.update_xaxes(tickfont=dict(size=12))
            fig.update_yaxes(tickfont=dict(size=12))
            
            # Add padding to y-axes ranges for bar charts
            fig.update_yaxes(range=[0, max(var_values) * 1.2], row=1, col=1)
            fig.update_yaxes(range=[0, max(ratio_values) * 1.2], row=1, col=2)
            fig.update_yaxes(range=[0, max(downside_values) * 1.2], row=2, col=1)
            
            # Update scatter plot axes
            fig.update_xaxes(title_text="Volatility (%)", title_font=dict(size=12), row=1, col=3)
            fig.update_yaxes(title_text="Expected Return (%)", title_font=dict(size=12), row=1, col=3)
            
            # Add gridlines for better readability
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating risk dashboard: {e}")
            return go.Figure()
    
    @staticmethod
    def create_var_distribution_chart(returns: np.ndarray, var_95: float, var_99: float) -> go.Figure:
        """Create VaR distribution visualization"""
        try:
            if len(returns) == 0:
                return go.Figure()
            
            fig = go.Figure()
            
            # Create histogram of returns
            fig.add_trace(go.Histogram(
                x=returns * 100,
                nbinsx=50,
                name='Return Distribution',
                opacity=0.7,
                marker_color='lightblue'
            ))
            
            # Add VaR lines
            fig.add_vline(
                x=var_95 * 100,
                line_dash="dash",
                line_color="red",
                annotation_text=f"VaR 95%: {var_95:.2%}"
            )
            
            fig.add_vline(
                x=var_99 * 100,
                line_dash="dash",
                line_color="darkred",
                annotation_text=f"VaR 99%: {var_99:.2%}"
            )
            
            fig.update_layout(
                title="Return Distribution with Value at Risk",
                xaxis_title="Daily Returns (%)",
                yaxis_title="Frequency",
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating VaR distribution chart: {e}")
            return go.Figure()
    
    @staticmethod
    def create_drawdown_chart(price_data: np.ndarray) -> go.Figure:
        """Create drawdown analysis chart"""
        try:
            if len(price_data) < 2:
                return go.Figure()
            
            # Calculate returns and cumulative performance
            returns = np.diff(price_data) / price_data[:-1]
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            
            dates = pd.date_range(end=datetime.now(), periods=len(drawdowns), freq='D')
            
            fig = go.Figure()
            
            # Add drawdown area
            fig.add_trace(go.Scatter(
                x=dates,
                y=drawdowns * 100,
                fill='tonexty',
                mode='lines',
                name='Drawdown',
                line_color='red',
                fillcolor='rgba(255,0,0,0.3)'
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            
            fig.update_layout(
                title="Drawdown Analysis",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating drawdown chart: {e}")
            return go.Figure()

def get_risk_analysis_data(ticker: str, prediction: Dict[str, Any]) -> Tuple[RiskMetrics, np.ndarray]:
    """
    Get risk analysis data for the specified ticker
    
    Args:
        ticker: Asset ticker symbol
        prediction: Current prediction dictionary
        
    Returns:
        Tuple of (RiskMetrics, price_data)
    """
    try:
        # Initialize risk analyzer
        risk_analyzer = AdvancedRiskAnalyzer()
        
        # Try to get real data first
        if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager:
            try:
                # Get historical data
                multi_tf_data = st.session_state.data_manager.fetch_multi_timeframe_data(ticker, ['1d'])
                
                if multi_tf_data and '1d' in multi_tf_data:
                    data = multi_tf_data['1d']
                    price_data = data['Close'].values[-252:]  # Last year of data
                    
                    # Calculate comprehensive risk metrics
                    risk_metrics = risk_analyzer.calculate_comprehensive_risk_metrics(
                        price_data, prediction
                    )
                    
                    return risk_metrics, price_data
            
            except Exception as e:
                logger.warning(f"Error getting real data for risk analysis: {e}")
        
        # Fallback to simulated data
        current_price = prediction.get('current_price', 100)
        predicted_price = prediction.get('predicted_price', current_price)
        
        # Generate realistic price history
        price_data = generate_simulated_price_data(ticker, current_price, 252)
        
        # Calculate risk metrics
        risk_metrics = risk_analyzer.calculate_comprehensive_risk_metrics(
            price_data, prediction
        )
        
        return risk_metrics, price_data
        
    except Exception as e:
        logger.error(f"Error getting risk analysis data: {e}")
        # Return fallback metrics
        risk_analyzer = AdvancedRiskAnalyzer()
        return risk_analyzer._generate_fallback_metrics(), np.array([100, 101, 99, 102, 98])

def generate_simulated_price_data(ticker: str, current_price: float, days: int) -> np.ndarray:
    """Generate realistic simulated price data for risk analysis"""
    try:
        from enhprog import get_asset_type, get_reasonable_price_range
        
        asset_type = get_asset_type(ticker)
        
        # Asset-specific volatility parameters
        volatility_params = {
            'crypto': 0.04,     # 4% daily volatility
            'forex': 0.008,     # 0.8% daily volatility
            'commodity': 0.025, # 2.5% daily volatility
            'index': 0.015,     # 1.5% daily volatility
            'stock': 0.02       # 2% daily volatility
        }
        
        daily_vol = volatility_params.get(asset_type, 0.02)
        daily_return = 0.0005  # Small positive drift
        
        # Generate price series using geometric Brownian motion
        np.random.seed(hash(ticker) % 1000)  # Consistent but ticker-specific seed
        
        returns = np.random.normal(daily_return, daily_vol, days)
        
        # Start from a reasonable historical price
        start_price = current_price * np.random.uniform(0.8, 1.2)
        
        # Generate price series
        prices = [start_price]
        for i in range(days - 1):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(new_price)
        
        # Ensure we end near the current price
        price_series = np.array(prices)
        adjustment_factor = current_price / price_series[-1]
        price_series *= adjustment_factor
        
        return price_series
        
    except Exception as e:
        logger.error(f"Error generating simulated price data: {e}")
        # Return simple random walk
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, days)
        prices = [current_price]
        for ret in returns[:-1]:
            prices.append(prices[-1] * (1 + ret))
        return np.array(prices)

def display_enhanced_risk_analysis_tab(prediction: Dict[str, Any]):
    """
    Main function to display enhanced risk analysis tab
    
    Args:
        prediction: Current prediction dictionary
    """
    st.subheader("⚠️ Advanced Risk Analysis")
    
    ticker = prediction.get('ticker', 'Unknown')
    current_price = prediction.get('current_price', 0)
    
    # Get risk analysis data
    with st.spinner("🔄 Calculating comprehensive risk metrics..."):
        risk_metrics, price_data = get_risk_analysis_data(ticker, prediction)
    
    # Display key risk metrics
    st.markdown("#### 🎯 Key Risk Metrics")
    
    risk_cols = st.columns(4)
    
    with risk_cols[0]:
        var_95_pct = abs(risk_metrics.var_95) * 100
        color = "🔴" if var_95_pct > 5 else "🟡" if var_95_pct > 3 else "🟢"
        st.metric(
            "VaR (95%)", 
            f"{var_95_pct:.2f}%",
            help="Maximum expected daily loss with 95% confidence"
        )
        st.markdown(f"{color} Risk Level")
    
    with risk_cols[1]:
        sharpe = risk_metrics.sharpe_ratio
        color = "🟢" if sharpe > 1.5 else "🟡" if sharpe > 1.0 else "🔴"
        st.metric(
            "Sharpe Ratio", 
            f"{sharpe:.2f}",
            help="Risk-adjusted return measure"
        )
        st.markdown(f"{color} Performance")
    
    with risk_cols[2]:
        volatility_pct = risk_metrics.volatility * 100
        color = "🔴" if volatility_pct > 40 else "🟡" if volatility_pct > 25 else "🟢"
        st.metric(
            "Volatility", 
            f"{volatility_pct:.1f}%",
            help="Annualized price volatility"
        )
        st.markdown(f"{color} Volatility")
    
    with risk_cols[3]:
        max_dd_pct = abs(risk_metrics.max_drawdown) * 100
        color = "🔴" if max_dd_pct > 20 else "🟡" if max_dd_pct > 10 else "🟢"
        st.metric(
            "Max Drawdown", 
            f"{max_dd_pct:.1f}%",
            help="Largest peak-to-trough decline"
        )
        st.markdown(f"{color} Drawdown Risk")
    
    # Comprehensive risk dashboard
    st.markdown("#### 📊 Risk Analysis Dashboard")
    
    risk_dashboard = RiskVisualization.create_risk_dashboard(risk_metrics, ticker)
    if risk_dashboard:
        st.plotly_chart(risk_dashboard, use_container_width=True)
    
    # Additional risk metrics
    st.markdown("#### 📈 Additional Risk Metrics")
    
    additional_cols = st.columns(4)
    
    with additional_cols[0]:
        st.metric("Sortino Ratio", f"{risk_metrics.sortino_ratio:.2f}", help="Downside risk-adjusted return")
    
    with additional_cols[1]:
        st.metric("Beta", f"{risk_metrics.beta:.2f}", help="Market sensitivity")
    
    with additional_cols[2]:
        st.metric("CVaR (95%)", f"{abs(risk_metrics.cvar_95)*100:.2f}%", help="Expected loss beyond VaR")
    
    with additional_cols[3]:
        st.metric("Information Ratio", f"{risk_metrics.information_ratio:.2f}", help="Risk-adjusted excess return")
    
    # Risk analysis tabs
    analysis_tabs = st.tabs(["📊 Distribution Analysis", "📉 Drawdown Analysis", "💰 Position Sizing", "📋 Risk Report"])
    
    with analysis_tabs[0]:
        st.markdown("##### Return Distribution with VaR")
        returns = np.diff(price_data) / price_data[:-1] if len(price_data) > 1 else np.array([])
        
        if len(returns) > 0:
            var_chart = RiskVisualization.create_var_distribution_chart(
                returns, risk_metrics.var_95, risk_metrics.var_99
            )
            st.plotly_chart(var_chart, use_container_width=True)
        else:
            st.warning("Insufficient data for distribution analysis")
    
    with analysis_tabs[1]:
        st.markdown("##### Historical Drawdown Analysis")
        
        if len(price_data) > 1:
            drawdown_chart = RiskVisualization.create_drawdown_chart(price_data)
            st.plotly_chart(drawdown_chart, use_container_width=True)
        else:
            st.warning("Insufficient data for drawdown analysis")
    
    with analysis_tabs[2]:
        st.markdown("##### Optimal Position Sizing")
        
        # Position sizing inputs
        pos_cols = st.columns(3)
        
        with pos_cols[0]:
            account_balance = st.number_input(
                "Account Balance ($)",
                min_value=1000,
                max_value=1000000,
                value=100000,
                step=5000
            )
        
        with pos_cols[1]:
            risk_per_trade = st.slider(
                "Risk per Trade (%)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.1
            ) / 100
        
        with pos_cols[2]:
            stop_loss_pct = st.slider(
                "Stop Loss (%)",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5
            ) / 100
        
        # Calculate position sizing
        risk_analyzer = AdvancedRiskAnalyzer()
        position_sizing = risk_analyzer.calculate_position_sizing(
            risk_metrics, account_balance, risk_per_trade, current_price, stop_loss_pct
        )
        
        # Display position sizing results
        st.markdown("##### Position Sizing Recommendations")
        
        sizing_cols = st.columns(4)
        
        with sizing_cols[0]:
            st.metric(
                "Recommended Shares", 
                f"{position_sizing['recommended_shares']:,}",
                help="Shares based on risk management"
            )
        
        with sizing_cols[1]:
            st.metric(
                "Position Value", 
                f"${position_sizing['position_value']:,.0f}",
                help="Total position value"
            )
        
        with sizing_cols[2]:
            st.metric(
                "Portfolio %", 
                f"{position_sizing['portfolio_pct']:.1f}%",
                help="Percentage of portfolio"
            )
        
        with sizing_cols[3]:
            st.metric(
                "Kelly Fraction", 
                f"{position_sizing['kelly_fraction']:.1%}",
                help="Kelly criterion recommendation"
            )
    
    with analysis_tabs[3]:
        st.markdown("##### Comprehensive Risk Report")
        
        # Generate and display risk report
        risk_analyzer = AdvancedRiskAnalyzer()
        risk_report = risk_analyzer.generate_risk_report(risk_metrics, ticker)
        
        st.text_area(
            "Detailed Risk Analysis",
            value=risk_report,
            height=400,
            help="Comprehensive risk analysis report"
        )
        
        # Download report option
        st.download_button(
            label="📥 Download Risk Report",
            data=risk_report,
            file_name=f"{ticker}_risk_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    # Risk warnings
    if risk_metrics.var_95 < -0.05:
        st.error("🚨 **HIGH RISK WARNING**: VaR exceeds 5% - Consider reducing position size")
    
    if risk_metrics.volatility > 0.4:
        st.warning("⚠️ **HIGH VOLATILITY**: Asset shows elevated volatility - Exercise caution")
    
    if risk_metrics.sharpe_ratio < 0.5:
        st.warning("⚠️ **POOR RISK-ADJUSTED RETURNS**: Sharpe ratio below acceptable threshold")