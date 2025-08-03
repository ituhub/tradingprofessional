import streamlit as st

def create_mobile_viewport_meta():
    """Add mobile-friendly viewport meta tag"""
    st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1, 
    maximum-scale=1, user-scalable=no">
    """, unsafe_allow_html=True)

def is_mobile_device():
    """
    Detect mobile device using multiple strategies
    
    Returns:
        bool: True if device is mobile, False otherwise
    """
    # First, check query parameters
    mobile_param = st.query_params.get('mobile', [False])[0]
    if mobile_param:
        return True
    
    # Add CSS media query detection
    st.markdown("""
    <style>
    @media (max-width: 768px) {
        body::before {
            content: 'mobile';
            display: none;
        }
    }
    </style>
    <script>
    // Detect device type using screen width
    function getDeviceType() {
        const width = window.innerWidth;
        return width <= 768;
    }
    
    // Store device type in a global variable
    window.isMobileDevice = getDeviceType();
    console.log('Is Mobile Device:', window.isMobileDevice);
    </script>
    """, unsafe_allow_html=True)
    
    return False

def create_mobile_layout():
    """Create a simplified, mobile-friendly layout"""
    st.markdown("""
    <style>
    /* Mobile-specific layout optimizations */
    @media (max-width: 768px) {
        /* Responsive page configuration */
        .stApp {
            padding: 10px !important;
            background: #f4f4f4 !important;
        }
        
        .main .block-container {
            padding: 0.5rem !important;
            margin: 0 !important;
        }
        
        /* Reduced font sizes */
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.3rem !important; }
        h3 { font-size: 1.1rem !important; }
        
        /* Columns adapt to mobile */
        .stColumns {
            display: flex;
            flex-direction: column;
        }
        
        /* Compact metrics and buttons */
        [data-testid="metric-container"] {
            padding: 0.5rem !important;
            margin-bottom: 10px !important;
        }
        
        .stButton > button {
            padding: 0.4rem 0.8rem !important;
            font-size: 0.9rem !important;
            min-height: 44px;
            min-width: 44px;
        }
        
        /* Sidebar adjustments */
        .css-1d391kg {
            width: 100% !important;
            padding: 10px !important;
        }
        
        /* Charts responsiveness */
        .js-plotly-plot {
            width: 100% !important;
            height: auto !important;
        }
        
        /* Tabs and interactive elements */
        .stTabs [data-baseweb="tab"] {
            padding: 5px 10px !important;
            font-size: 0.9rem !important;
        }
        
        /* Touch-friendly interactions */
        .stSelectbox, .stMultiSelect {
            min-height: 44px;
        }
        
        /* Compact metric display */
        [data-testid="metric-container"] {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def optimize_mobile_performance():
    """Reduce computational intensity for mobile devices"""
    st.markdown("""
    <style>
    /* Performance and usability optimizations */
    @media (max-width: 768px) {
        /* Reduce complex animations */
        .metric-card:hover {
            transform: none !important;
            box-shadow: none !important;
        }
        
        /* Simplified gradients */
        .stApp {
            background: #f4f4f4 !important;
        }
        
        /* Typography and readability */
        body {
            font-size: 14px !important;
            line-height: 1.5 !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def mobile_chart_optimization(fig):
    """
    Simplify chart for mobile devices
    
    Args:
        fig (plotly.graph_objs._figure.Figure): Plotly figure to modify
    
    Returns:
        plotly.graph_objs._figure.Figure: Optimized mobile chart
    """
    fig.update_layout(
        height=300,  # Reduced height
        margin=dict(l=10, r=10, t=30, b=10),  # Compact margins
        showlegend=False  # Optional: hide legend to save space
    )
    return fig

def apply_mobile_optimizations():
    """
    Apply comprehensive mobile optimizations
    Call this after set_page_config
    """
    # Add mobile viewport meta
    create_mobile_viewport_meta()
    
    # Detect if it's a mobile device
    is_mobile = is_mobile_device()
    
    # Apply mobile-specific optimizations if detected
    if is_mobile:
        create_mobile_layout()
        optimize_mobile_performance()
        
        # Optional: Show mobile optimization warning
        st.warning("🚀 Mobile Optimized View")
    
    return is_mobile

def get_device_type():
    """
    Comprehensive device type detection
    
    Returns:
        str: 'mobile', 'tablet', or 'desktop'
    """
    # Check query parameters
    mobile_param = st.query_params.get('mobile', [False])[0]
    if mobile_param:
        return 'mobile'
    
    # Add more sophisticated detection
    st.markdown("""
    <script>
    // Detect device type
    function getDeviceType() {
        const width = window.innerWidth;
        if (width <= 600) return 'mobile';
        if (width <= 1024) return 'tablet';
        return 'desktop';
    }
    
    // Store device type
    window.deviceType = getDeviceType();
    console.log('Detected Device:', window.deviceType);
    </script>
    """, unsafe_allow_html=True)
    
    return 'desktop'  # Default fallback