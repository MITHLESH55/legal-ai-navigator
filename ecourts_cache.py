"""
E-Courts Cache Helper

Provides functions to load cached e-Courts static data.
Falls back to API if cache is missing or expired.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_TTL_DAYS = 7

CACHE_FILES = {
    "states": CACHE_DIR / "ecourts_states.json",
    "districts": CACHE_DIR / "ecourts_districts.json",
    "courts": CACHE_DIR / "ecourts_courts.json",
    "consumer_districts": CACHE_DIR / "consumer_forum_districts.json",
    "consumer_benches": CACHE_DIR / "consumer_forum_benches.json",
    "metadata": CACHE_DIR / "cache_metadata.json"
}


def is_cache_valid() -> bool:
    """Check if cache exists and is not expired"""
    metadata_file = CACHE_FILES["metadata"]
    
    if not metadata_file.exists():
        return False
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        cached_date = datetime.fromisoformat(metadata["cached_at"])
        expiry_date = cached_date + timedelta(days=CACHE_TTL_DAYS)
        
        return datetime.now() <= expiry_date
    except Exception as e:
        logger.error(f"Error reading cache metadata: {e}")
        return False


def load_from_cache(cache_key: str) -> Optional[Dict]:
    """
    Load data from cache file
    
    Args:
        cache_key: One of 'states', 'districts', 'courts', 'consumer_districts', 'consumer_benches'
    
    Returns:
        Cached data or None if cache doesn't exist/is invalid
    """
    if not is_cache_valid():
        logger.warning(f"Cache is invalid or expired. Run: python cache_ecourts_data.py")
        return None
    
    cache_file = CACHE_FILES.get(cache_key)
    if not cache_file or not cache_file.exists():
        logger.warning(f"Cache file not found: {cache_key}")
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading cache {cache_key}: {e}")
        return None


def get_states_cached() -> Optional[List[Dict]]:
    """Get cached states list"""
    return load_from_cache("states")


def get_districts_cached(state_code: Optional[str] = None) -> Optional[Dict]:
    """
    Get cached districts
    
    Args:
        state_code: If provided, returns only districts for that state
    
    Returns:
        All districts dict or specific state's districts
    """
    districts_data = load_from_cache("districts")
    
    if not districts_data:
        return None
    
    if state_code:
        return districts_data.get(state_code)
    
    return districts_data


def get_courts_cached(state_code: Optional[str] = None, district_code: Optional[str] = None) -> Optional[Dict]:
    """
    Get cached courts
    
    Args:
        state_code: If provided, returns courts for that state
        district_code: If provided (with state_code), returns courts for that district
    
    Returns:
        Courts data
    """
    courts_data = load_from_cache("courts")
    
    if not courts_data:
        return None
    
    if state_code:
        state_courts = courts_data.get(state_code)
        if not state_courts:
            return None
        
        if district_code:
            return state_courts.get("districts", {}).get(district_code)
        
        return state_courts
    
    return courts_data


def get_consumer_districts_cached() -> Optional[List[Dict]]:
    """Get cached consumer forum districts"""
    return load_from_cache("consumer_districts")


def get_consumer_benches_cached() -> Optional[List[Dict]]:
    """Get cached consumer forum benches"""
    return load_from_cache("consumer_benches")


def get_cache_info() -> Dict:
    """Get information about cache status"""
    metadata_file = CACHE_FILES["metadata"]
    
    if not metadata_file.exists():
        return {
            "cached": False,
            "message": "No cache found. Run: python cache_ecourts_data.py"
        }
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        cached_date = datetime.fromisoformat(metadata["cached_at"])
        expiry_date = datetime.fromisoformat(metadata["expires_at"])
        is_valid = datetime.now() <= expiry_date
        
        return {
            "cached": True,
            "valid": is_valid,
            "cached_at": metadata["cached_at"],
            "expires_at": metadata["expires_at"],
            "ttl_days": metadata["ttl_days"],
            "files": metadata.get("files", {}),
            "message": "Cache is valid" if is_valid else "Cache expired. Run: python cache_ecourts_data.py --force"
        }
    except Exception as e:
        return {
            "cached": False,
            "error": str(e),
            "message": "Error reading cache. Run: python cache_ecourts_data.py --force"
        }
