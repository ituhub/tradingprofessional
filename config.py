import os
import secrets
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Application Configuration
class AppConfig:
    # API Keys
    FMP_API_KEY = os.getenv('FMP_API_KEY')
    FRED_API_KEY = os.getenv('FRED_API_KEY')
    
    # Security Settings
    SECRET_KEY = os.getenv('APP_SECRET_KEY', secrets.token_hex(32))
    
    # Rate Limiting
    MAX_PREDICTIONS_FREE = 10
    MAX_PREDICTIONS_PREMIUM = 1000
    
    # Subscription Tiers
    TIER_FEATURES = {
        'free': {
            'max_predictions': 10,
            'max_models': 2,
            'available_features': ['basic_prediction']
        },
        'premium': {
            'max_predictions': 1000,
            'max_models': 8,
            'available_features': ['all_features']
        }
    }

    @classmethod
    def get_tier_features(cls, tier):
        return cls.TIER_FEATURES.get(tier, cls.TIER_FEATURES['free'])

# Privacy Policy
PRIVACY_POLICY = """
# AI Trading Professional - Privacy Policy

## Data Collection
- We collect minimal user information required for account management
- No financial data is stored permanently
- API interactions are logged for security purposes

## Data Usage
- User data is used solely for authentication and service provision
- We do not sell or share personal information

## User Rights
- Users can request account deletion at any time
- Data retention is limited to active account periods

## Compliance
- GDPR and CCPA compliant
- Secure data handling practices
"""