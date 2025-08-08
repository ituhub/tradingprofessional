"""
Premium Keys Management Module for AI Trading Professional

This module defines custom premium keys with specific access levels and usage limitations.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta

class PremiumKey:
    """
    Advanced Premium Key Management Class
    
    Allows fine-grained control over premium features and usage tracking
    """
    
    def __init__(
        self, 
        key: str, 
        name: str, 
        max_predictions: int = float('inf'), 
        features: List[str] = None,
        expiration_days: int = 30,
        allow_model_management: bool = False
    ):
        """
        Initialize a premium key with specific characteristics
        
        Args:
            key (str): The unique premium key
            name (str): Name or description of the key
            max_predictions (int): Maximum number of predictions allowed
            features (List[str]): List of enabled features
            expiration_days (int): Number of days the key remains valid
            allow_model_management (bool): Whether model management is allowed
        """
        self.key = key
        self.name = name
        self.max_predictions = max_predictions
        self.allow_model_management = allow_model_management
        
        # Default features if not specified
        self.features = features or [
            'Real-time Predictions',
            'Cross-Validation',
            'Advanced Analytics', 
            'Portfolio Management',
            'Backtesting',
            'Alternative Data',
            'Advanced Risk Analysis',
            'Regime Detection',
            'Drift Detection',
            'SHAP Model Explanations'
        ]
        
        # Conditionally add Model Management feature
        if allow_model_management and 'Model Management' not in self.features:
            self.features.append('Model Management')
        elif not allow_model_management:
            # Ensure Model Management is removed
            self.features = [f for f in self.features if f != 'Model Management']
        
        # Usage tracking
        self.activation_timestamp = datetime.now()
        self.expiration_timestamp = self.activation_timestamp + timedelta(days=expiration_days)
        self.prediction_count = 0
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate the premium key and check its current status
        
        Returns:
            Dict containing validation details
        """
        current_time = datetime.now()
        
        # Check expiration
        if current_time > self.expiration_timestamp:
            return {
                'valid': False,
                'tier': 'expired',
                'message': 'Premium key has expired',
                'features': []
            }
        
        # Check prediction limit
        if self.prediction_count >= self.max_predictions:
            return {
                'valid': False,
                'tier': 'exhausted',
                'message': 'Maximum predictions reached',
                'features': []
            }
        
        # Valid key response
        return {
            'valid': True,
            'tier': 'premium',
            'key': self.key,
            'name': self.name,
            'features': self.features,
            'predictions_remaining': (float('inf') if self.max_predictions == float('inf') else self.max_predictions - self.prediction_count),
            'expires': self.expiration_timestamp.isoformat(),
            'activation_date': self.activation_timestamp.isoformat(),
            'allow_model_management': self.allow_model_management,
            'message': f'Welcome to {self.name} Premium Access!'
        }
    
    def use_prediction(self) -> bool:
        """
        Increment prediction count if key is valid
        
        Returns:
            bool: Whether a prediction can be generated
        """
        validation = self.validate()
        
        if validation['valid']:
            # Only increment for keys with finite predictions
            if self.max_predictions != float('inf'):
                self.prediction_count += 1
            return True
        
        return False
    
    def reset_predictions(self):
        """
        Reset prediction count (useful for key renewal)
        """
        self.prediction_count = 0
        self.activation_timestamp = datetime.now()
        self.expiration_timestamp = self.activation_timestamp + timedelta(days=30)

# Predefined Premium Keys
PREMIUM_KEYS = {
    'Prem246_357': PremiumKey(
        key='Prem246_357', 
        name='Professional Tier',
        max_predictions=float('inf'),  # Unlimited predictions
        features=None,  # All features
        allow_model_management=True  # Enable Model Management
    ),
    'Alpha_Trade_246': PremiumKey(
        key='Alpha_Trade_246', 
        name='Limited Pro Tier',
        max_predictions=15,  # 15 prediction limit
        features=None,  # All features
        allow_model_management=False  # Disable Model Management
    )
}

def validate_premium_key(key: str) -> Dict[str, Any]:
    """
    Global function to validate a premium key
    
    Args:
        key (str): Premium key to validate
    
    Returns:
        Dict with validation results
    """
    if key in PREMIUM_KEYS:
        return PREMIUM_KEYS[key].validate()
    
    return {
        'valid': False,
        'tier': 'invalid',
        'message': 'Unrecognized premium key',
        'features': []
    }

def use_premium_prediction(key: str) -> bool:
    """
    Attempt to use a prediction with a specific premium key
    
    Args:
        key (str): Premium key to use
    
    Returns:
        bool: Whether prediction can be generated
    """
    if key in PREMIUM_KEYS:
        return PREMIUM_KEYS[key].use_prediction()
    
    return False
"""
Premium Keys Management Module for AI Trading Professional

This module defines custom premium keys with specific access levels and usage limitations.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta

class PremiumKey:
    """
    Advanced Premium Key Management Class
    
    Allows fine-grained control over premium features and usage tracking
    """
    
    def __init__(
        self, 
        key: str, 
        name: str, 
        max_predictions: int = 15, 
        features: List[str] = None,
        expiration_days: int = 30
    ):
        """
        Initialize a premium key with specific characteristics
        
        Args:
            key (str): The unique premium key
            name (str): Name or description of the key
            max_predictions (int): Maximum number of predictions allowed
            features (List[str]): List of enabled features
            expiration_days (int): Number of days the key remains valid
        """
        self.key = key
        self.name = name
        self.max_predictions = max_predictions
        
        # Default features if not specified
        self.features = features or [
            'Real-time Predictions',
            'Cross-Validation',
            'Advanced Analytics', 
            'Portfolio Management',
            'Backtesting',
            'Alternative Data',
            'Advanced Risk Analysis',
            'Regime Detection',
            'Drift Detection',
            'SHAP Model Explanations'
        ]
        
        # Explicitly remove Model Management feature
        if 'Model Management' in self.features:
            self.features.remove('Model Management')
        
        # Ensure Model Management is completely excluded
        self.features = [f for f in self.features if f != 'Model Management']
        
        # Usage tracking
        self.activation_timestamp = datetime.now()
        self.expiration_timestamp = self.activation_timestamp + timedelta(days=expiration_days)
        self.prediction_count = 0
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate the premium key and check its current status
        
        Returns:
            Dict containing validation details
        """
        current_time = datetime.now()
        
        # Check expiration
        if current_time > self.expiration_timestamp:
            return {
                'valid': False,
                'tier': 'expired',
                'message': 'Premium key has expired',
                'features': []
            }
        
        # Check prediction limit
        if self.prediction_count >= self.max_predictions:
            return {
                'valid': False,
                'tier': 'exhausted',
                'message': 'Maximum predictions reached',
                'features': []
            }
        
        # Valid key response
        return {
            'valid': True,
            'tier': 'premium',
            'key': self.key,
            'name': self.name,
            'features': self.features,
            'predictions_remaining': self.max_predictions - self.prediction_count,
            'expires': self.expiration_timestamp.isoformat(),
            'activation_date': self.activation_timestamp.isoformat(),
            'message': f'Welcome to {self.name} Premium Access!'
        }
    
    def use_prediction(self) -> bool:
        """
        Increment prediction count if key is valid
        
        Returns:
            bool: Whether a prediction can be generated
        """
        validation = self.validate()
        
        if validation['valid']:
            self.prediction_count += 1
            return True
        
        return False
    
    def reset_predictions(self):
        """
        Reset prediction count (useful for key renewal)
        """
        self.prediction_count = 0
        self.activation_timestamp = datetime.now()
        self.expiration_timestamp = self.activation_timestamp + timedelta(days=30)

# Predefined Premium Keys
PREMIUM_KEYS = {
    'Prem246_357': PremiumKey(
        key='Prem246_357', 
        name='Professional Tier',
        max_predictions=float('inf'),  # Unlimited predictions
        features=None  # All features
    ),
    'Alpha_Trade_246': PremiumKey(
        key='Alpha_Trade_246', 
        name='Limited Pro Tier',
        max_predictions=15,  # 15 prediction limit
        features=None  # All features except model management
    )
}

def validate_premium_key(key: str) -> Dict[str, Any]:
    """
    Global function to validate a premium key
    
    Args:
        key (str): Premium key to validate
    
    Returns:
        Dict with validation results
    """
    if key in PREMIUM_KEYS:
        return PREMIUM_KEYS[key].validate()
    
    return {
        'valid': False,
        'tier': 'invalid',
        'message': 'Unrecognized premium key',
        'features': []
    }

def use_premium_prediction(key: str) -> bool:
    """
    Attempt to use a prediction with a specific premium key
    
    Args:
        key (str): Premium key to use
    
    Returns:
        bool: Whether prediction can be generated
    """
    if key in PREMIUM_KEYS:
        return PREMIUM_KEYS[key].use_prediction()
    
    return False