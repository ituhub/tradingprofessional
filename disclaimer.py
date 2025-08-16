"""
INVESTMENT DISCLAIMER AND RISK WARNING MODULE
===========================================
Comprehensive disclaimer system for AI Trading Professional
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any


class InvestmentDisclaimer:
    """
    Comprehensive investment disclaimer and risk warning system
    """
    
    @staticmethod
    def display_disclaimer() -> bool:
        """
        Display comprehensive investment disclaimer and handle consent.
        
        Returns:
            bool: True if user consented, False otherwise
        """
        # Use session state to track disclaimer consent
        if 'disclaimer_consented' not in st.session_state:
            st.session_state.disclaimer_consented = False
        
        if st.session_state.disclaimer_consented:
            return True
        
        # Display the disclaimer
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff6b6b, #ee5a52); 
                    color: white; padding: 20px; border-radius: 10px; 
                    box-shadow: 0 4px 15px rgba(255,107,107,0.3); margin-bottom: 20px;">
            <h2 style="color: white; text-align: center; margin: 0;">
                🚨 CRITICAL INVESTMENT RISK WARNING
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## ⚠️ PLEASE READ CAREFULLY BEFORE PROCEEDING")
        
        # Main disclaimer content
        st.markdown("""
        ### 📋 IMPORTANT LEGAL DISCLAIMERS
        
        **By using this AI Trading Professional platform, you explicitly acknowledge and agree to the following:**
        
        ---
        
        #### 🤖 1. ALGORITHMIC PREDICTIONS DISCLAIMER
        - **NOT INVESTMENT ADVICE**: All predictions are algorithmic outputs based on historical data
        - **NO GUARANTEES**: Past performance does NOT guarantee future results
        - **INFORMATIONAL ONLY**: Platform provides analysis tools, not investment recommendations
        - **MODEL LIMITATIONS**: AI models cannot predict black swan events or unprecedented market conditions
        
        #### 💸 2. FINANCIAL RISK WARNING
        - **CAPITAL LOSS RISK**: You may lose some or ALL of your invested capital
        - **VOLATILITY RISK**: Markets can be extremely volatile and unpredictable
        - **LEVERAGE RISK**: Trading with leverage amplifies both gains and losses
        - **LIQUIDITY RISK**: Some assets may be difficult to buy or sell quickly
        
        #### 📈 3. MARKET RISKS
        - **SYSTEMATIC RISK**: Entire markets can decline due to economic factors
        - **CURRENCY RISK**: Foreign exchange fluctuations can affect returns
        - **INTEREST RATE RISK**: Changes in interest rates can impact asset values
        - **POLITICAL RISK**: Government actions and regulations can affect markets
        
        #### 🎯 4. TRADING SPECIFIC RISKS
        - **EXECUTION RISK**: Orders may not execute at expected prices
        - **SLIPPAGE RISK**: Actual execution prices may differ from quoted prices
        - **GAP RISK**: Markets can open significantly different from previous close
        - **TECHNOLOGY RISK**: System failures or connectivity issues can impact trading
        
        #### 🧠 5. AI AND TECHNOLOGY LIMITATIONS
        - **DATA DEPENDENCY**: AI models depend on quality and availability of data
        - **OVERFITTING RISK**: Models may perform well on historical data but fail in real conditions
        - **BIAS RISK**: AI models may contain inherent biases from training data
        - **BLACK BOX RISK**: Complex AI decisions may not always be fully explainable
        
        #### 👤 6. PERSONAL RESPONSIBILITY
        - **SOLE RESPONSIBILITY**: YOU are solely responsible for ALL investment decisions
        - **DUE DILIGENCE**: You must conduct your own research and analysis
        - **RISK TOLERANCE**: Only invest what you can afford to lose completely
        - **PROFESSIONAL ADVICE**: Consider consulting with qualified financial advisors
        
        #### ⚖️ 7. LEGAL AND REGULATORY
        - **NO FIDUCIARY DUTY**: Platform operators have no fiduciary relationship with users
        - **REGULATORY COMPLIANCE**: You are responsible for compliance with local regulations
        - **TAX IMPLICATIONS**: You are responsible for understanding tax consequences
        - **JURISDICTION**: Platform may not be suitable for users in all jurisdictions
        
        ---
        
        ### 🔴 ADDITIONAL WARNINGS
        
        #### Cryptocurrency Trading
        - Extremely high volatility and regulatory uncertainty
        - Risk of total loss due to technological or regulatory changes
        - Limited regulatory protection compared to traditional assets
        
        #### Forex Trading
        - High leverage can result in losses exceeding initial deposits
        - 24-hour markets with gaps and sudden price movements
        - Central bank interventions can cause dramatic price changes
        
        #### Commodities Trading
        - Prices affected by weather, geopolitical events, and supply disruptions
        - Storage costs and delivery obligations in physical markets
        - Seasonal patterns may not repeat consistently
        
        ---
        
        ### 📊 PERFORMANCE DISCLAIMERS
        
        - **HYPOTHETICAL RESULTS**: Backtesting results are hypothetical and may not reflect actual trading
        - **TRANSACTION COSTS**: Real trading involves commissions, spreads, and slippage not fully captured in simulations
        - **MARKET CONDITIONS**: Past market conditions may not repeat in the future
        - **SURVIVORSHIP BIAS**: Results may not account for failed strategies or discontinued assets
        
        ---
        
        ### 🛡️ PLATFORM DISCLAIMERS
        
        - **NO WARRANTY**: Platform provided "as is" without warranties of any kind
        - **AVAILABILITY**: Platform may experience downtime or technical issues
        - **DATA ACCURACY**: While we strive for accuracy, data may contain errors or delays
        - **FEATURE CHANGES**: Platform features and functionality may change without notice
        
        ---
        
        ### 📞 SEEKING PROFESSIONAL HELP
        
        **Consider consulting qualified professionals for:**
        - Investment strategy and portfolio construction
        - Tax planning and implications
        - Legal and regulatory compliance
        - Risk management and insurance needs
        
        ---
        """)
        
        # Consent section with enhanced styling
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                    color: white; padding: 25px; border-radius: 15px; 
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1); margin: 30px 0;">
            <h3 style="color: white; text-align: center; margin-bottom: 20px;">
                ⚖️ CONSENT REQUIRED TO PROCEED
            </h3>
            <p style="text-align: center; font-size: 16px; margin: 0;">
                You must acknowledge that you have read, understood, and agree to all the above terms and risks.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Consent buttons
        consent_col1, consent_col2 = st.columns(2)
        
        with consent_col1:
            if st.button(
                "✅ I HAVE READ AND CONSENT TO ALL TERMS",
                type="primary",
                key="disclaimer_consent_button",
                help="Click to acknowledge all risks and proceed to the platform",
                use_container_width=True
            ):
                st.session_state.disclaimer_consented = True
                st.session_state.disclaimer_consent_timestamp = datetime.now().isoformat()
                st.success("✅ Consent recorded. You may now proceed to use the platform.")
                st.rerun()
        
        with consent_col2:
            if st.button(
                "❌ I DO NOT CONSENT",
                type="secondary",
                key="disclaimer_decline_button",
                help="Click if you do not agree to the terms - this will prevent platform access",
                use_container_width=True
            ):
                st.session_state.disclaimer_consented = False
                st.error("❌ **ACCESS DENIED** - You must consent to the terms to use this platform.")
                st.info("If you change your mind, please refresh the page and review the terms again.")
                st.stop()
        
        return False
    
    @staticmethod
    def show_consent_status():
        """
        Show current consent status in a compact format
        """
        if st.session_state.get('disclaimer_consented', False):
            consent_time = st.session_state.get('disclaimer_consent_timestamp', 'Unknown')
            try:
                # Parse the timestamp and format it
                consent_dt = datetime.fromisoformat(consent_time)
                formatted_time = consent_dt.strftime('%Y-%m-%d %H:%M')
            except:
                formatted_time = consent_time
            
            st.markdown(
                f'<div style="background: #d4edda; color: #155724; padding: 10px; '
                f'border-radius: 5px; font-size: 12px; text-align: center; margin: 10px 0;">'
                f'✅ <strong>Risks Acknowledged</strong><br>'
                f'<small>Consented: {formatted_time}</small>'
                f'</div>',
                unsafe_allow_html=True
            )
    
    @staticmethod
    def reset_consent():
        """
        Reset consent status (for testing or administrative purposes)
        """
        if 'disclaimer_consented' in st.session_state:
            del st.session_state.disclaimer_consented
        if 'disclaimer_consent_timestamp' in st.session_state:
            del st.session_state.disclaimer_consent_timestamp
    
    @staticmethod
    def get_consent_data() -> Dict[str, Any]:
        """
        Get consent data for logging/auditing purposes
        
        Returns:
            Dict containing consent status and timestamp
        """
        return {
            'consented': st.session_state.get('disclaimer_consented', False),
            'timestamp': st.session_state.get('disclaimer_consent_timestamp', None),
            'user_agent': 'streamlit_app',  # Could be expanded with actual user agent
            'session_id': id(st.session_state)  # Simple session identifier
        }


class DisclaimerValidator:
    """
    Additional validation and compliance checking
    """
    
    @staticmethod
    def validate_consent_age() -> bool:
        """
        Check if consent is still valid (e.g., not too old)
        """
        if not st.session_state.get('disclaimer_consented', False):
            return False
        
        consent_time = st.session_state.get('disclaimer_consent_timestamp')
        if not consent_time:
            return False
        
        try:
            consent_dt = datetime.fromisoformat(consent_time)
            # Consider consent valid for 24 hours
            time_diff = datetime.now() - consent_dt
            return time_diff.total_seconds() < 86400  # 24 hours
        except:
            return False
    
    @staticmethod
    def require_fresh_consent_if_needed():
        """
        Force fresh consent if the existing one is too old
        """
        if not DisclaimerValidator.validate_consent_age():
            st.session_state.disclaimer_consented = False
            if 'disclaimer_consent_timestamp' in st.session_state:
                del st.session_state.disclaimer_consent_timestamp