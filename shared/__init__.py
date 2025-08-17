"""
Shared modules for the AI Trading Professional system
"""

__version__ = "1.0.0"
__author__ = "AI Trading Professional Team"

# Make imports available at package level
try:
    from .user_database import UserDatabase
    __all__ = ['UserDatabase']
except ImportError as e:
    print(f"Warning: Could not import UserDatabase: {e}")
    __all__ = []

# Optional: Add other shared imports here as your system grows
try:
    from .disclaimer import InvestmentDisclaimer, DisclaimerValidator
    __all__.extend(['InvestmentDisclaimer', 'DisclaimerValidator'])
except ImportError:
    # Disclaimer module is optional
    pass

try:
    from .premium_keys import validate_premium_access, use_premium_prediction
    __all__.extend(['validate_premium_access', 'use_premium_prediction'])
except ImportError:
    # Premium keys module is optional
    pass

# System information
SYSTEM_NAME = "AI Trading Professional"
DATABASE_VERSION = "1.0"
SUPPORTED_TIERS = ['free', 'tier_10', 'tier_25', 'tier_50', 'tier_100']

def get_system_info():
    """Get basic system information"""
    return {
        'name': SYSTEM_NAME,
        'version': __version__,
        'database_version': DATABASE_VERSION,
        'supported_tiers': SUPPORTED_TIERS,
        'available_modules': __all__
    }
