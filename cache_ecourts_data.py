"""
E-Courts Static Data Cache Manager

Fetches and caches static data from e-Courts API:
- States
- Districts (per state)
- Courts (per district)
- Consumer Forum districts
- Consumer Forum benches

Cache expires after 7 days (configurable TTL).

Usage:
    python cache_ecourts_data.py          # Fetch and cache all data
    python cache_ecourts_data.py --force  # Force refresh even if cache is valid
"""

import os
import json
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_TTL_DAYS = 7  # Cache expires after 7 days
ECOURTS_API_BASE = "https://court-api.kleopatra.io/api"
ECOURTS_API_TOKEN = os.getenv("ECOURTS_API_TOKEN")

# Cache file paths
CACHE_FILES = {
    "states": CACHE_DIR / "ecourts_states.json",
    "districts": CACHE_DIR / "ecourts_districts.json",
    "courts": CACHE_DIR / "ecourts_courts.json",
    "consumer_districts": CACHE_DIR / "consumer_forum_districts.json",
    "consumer_benches": CACHE_DIR / "consumer_forum_benches.json",
    "metadata": CACHE_DIR / "cache_metadata.json"
}


def ensure_cache_dir():
    """Create cache directory if it doesn't exist"""
    CACHE_DIR.mkdir(exist_ok=True)
    logger.info(f"Cache directory: {CACHE_DIR}")


def is_cache_valid() -> bool:
    """Check if cache exists and is not expired"""
    metadata_file = CACHE_FILES["metadata"]
    
    if not metadata_file.exists():
        logger.info("No cache metadata found")
        return False
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        cached_date = datetime.fromisoformat(metadata["cached_at"])
        expiry_date = cached_date + timedelta(days=CACHE_TTL_DAYS)
        
        if datetime.now() > expiry_date:
            logger.info(f"Cache expired (cached: {cached_date}, expiry: {expiry_date})")
            return False
        
        logger.info(f"Cache is valid (expires: {expiry_date})")
        return True
        
    except Exception as e:
        logger.error(f"Error reading cache metadata: {e}")
        return False


def fetch_states() -> List[Dict]:
    """Fetch list of all states from e-Courts API"""
    logger.info("Fetching states from e-Courts API...")
    
    url = f"{ECOURTS_API_BASE}/core/static/district-court/states"
    headers = {
        "authorization": f"Bearer {ECOURTS_API_TOKEN}"
    }
    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    states = response.json()
    if not isinstance(states, list):
        states = states.get("states", [])
    
    # Debug: Print first state to see structure
    if states and len(states) > 0:
        logger.info(f"DEBUG: First state structure: {states[0]}")
    
    logger.info(f"✅ Fetched {len(states)} states")
    return states


def fetch_all_districts_at_once() -> Dict:
    """Fetch all districts at once using the all parameter"""
    logger.info("Fetching all districts in one request...")
    
    url = f"{ECOURTS_API_BASE}/core/static/district-court/districts"
    headers = {
        "authorization": f"Bearer {ECOURTS_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, headers=headers, json={"all": True}, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    logger.info(f"✅ Fetched districts data")
    return data


def fetch_all_districts() -> Dict[str, List[Dict]]:
    """Fetch districts for all states"""
    logger.info("Fetching districts for all states...")
    
    # Load cached states to map state IDs to names
    with open(CACHE_FILES["states"], 'r') as f:
        states = json.load(f)
    
    # Create a map of state_id -> state_name
    state_map = {state["id"]: state["name"] for state in states}
    
    # Fetch all districts at once
    districts_data = fetch_all_districts_at_once()
    
    # Reorganize data with state names
    all_districts = {}
    for state_id, districts in districts_data.items():
        state_name = state_map.get(state_id, state_id)
        all_districts[state_id] = {
            "state_name": state_name,
            "state_id": state_id,
            "districts": districts
        }
        logger.info(f"  ✓ {state_name}: {len(districts)} districts")
    
    logger.info(f"✅ Fetched districts for {len(all_districts)} states")
    return all_districts
    
    logger.info(f"✅ Fetched districts for {len(all_districts)} states")
    return all_districts


def fetch_courts(state_code: str, district_code: str) -> List[Dict]:
    """Fetch courts for a specific district"""
    url = f"{ECOURTS_API_BASE}/core/static/district-court/courts"
    headers = {
        "authorization": f"Bearer {ECOURTS_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, headers=headers, json={
        "state_code": state_code,
        "district_code": district_code
    }, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    return data.get("courts", [])


def fetch_all_courts() -> Dict[str, Dict]:
    """Fetch courts for all districts (WARNING: This can take a while!)"""
    logger.info("Fetching courts for all districts...")
    logger.warning("This may take several minutes due to API rate limits...")
    
    # Load cached districts
    with open(CACHE_FILES["districts"], 'r') as f:
        districts_data = json.load(f)
    
    all_courts = {}
    
    for state_code, state_info in districts_data.items():
        state_name = state_info["state_name"]
        districts = state_info.get("districts", [])
        
        logger.info(f"  Processing {state_name} ({len(districts)} districts)...")
        
        state_courts = {}
        for district in districts:
            district_code = district.get("district_code")
            district_name = district.get("district_name")
            
            try:
                courts = fetch_courts(state_code, district_code)
                state_courts[district_code] = {
                    "district_name": district_name,
                    "district_code": district_code,
                    "courts": courts
                }
                logger.info(f"    ✅ {district_name}: {len(courts)} courts")
            except Exception as e:
                logger.error(f"    ❌ Error fetching courts for {district_name}: {e}")
                state_courts[district_code] = {
                    "district_name": district_name,
                    "district_code": district_code,
                    "courts": [],
                    "error": str(e)
                }
        
        all_courts[state_code] = {
            "state_name": state_name,
            "state_code": state_code,
            "districts": state_courts
        }
    
    logger.info(f"✅ Fetched courts for all states")
    return all_courts


def fetch_consumer_forum_districts() -> List[Dict]:
    """Fetch consumer forum districts"""
    logger.info("Fetching consumer forum districts...")
    
    url = f"{ECOURTS_API_BASE}/core/static/consumer-forum/districts"
    headers = {
        "authorization": f"Bearer {ECOURTS_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, headers=headers, json={"all": True}, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    districts = data if isinstance(data, list) else data.get("districts", [])
    
    logger.info(f"✅ Fetched {len(districts)} consumer forum districts")
    return districts


def fetch_consumer_forum_benches() -> List[Dict]:
    """Fetch consumer forum benches"""
    logger.info("Fetching consumer forum benches...")
    
    url = f"{ECOURTS_API_BASE}/core/static/consumer-forum/benches"
    headers = {
        "authorization": f"Bearer {ECOURTS_API_TOKEN}"
    }
    
    # This is a GET request, not POST
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    benches = data if isinstance(data, list) else data.get("benches", [])
    
    logger.info(f"✅ Fetched {len(benches)} consumer forum benches")
    return benches


def save_cache(data: Dict, cache_key: str):
    """Save data to cache file"""
    cache_file = CACHE_FILES[cache_key]
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved to {cache_file}")


def update_metadata():
    """Update cache metadata with current timestamp"""
    metadata = {
        "cached_at": datetime.now().isoformat(),
        "ttl_days": CACHE_TTL_DAYS,
        "expires_at": (datetime.now() + timedelta(days=CACHE_TTL_DAYS)).isoformat(),
        "files": {key: str(path) for key, path in CACHE_FILES.items() if key != "metadata"}
    }
    
    with open(CACHE_FILES["metadata"], 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"💾 Cache metadata updated (expires: {metadata['expires_at']})")


def cache_all_data(force: bool = False, include_courts: bool = False):
    """
    Fetch and cache all e-Courts static data
    
    Args:
        force: Force refresh even if cache is valid
        include_courts: Fetch courts data (WARNING: slow!)
    """
    ensure_cache_dir()
    
    if not force and is_cache_valid():
        logger.info("✅ Cache is still valid. Use --force to refresh.")
        return
    
    if not ECOURTS_API_TOKEN:
        logger.error("❌ ECOURTS_API_TOKEN not found in environment variables!")
        return
    
    logger.info("=" * 60)
    logger.info("  E-COURTS DATA CACHING")
    logger.info("=" * 60)
    logger.info(f"Cache TTL: {CACHE_TTL_DAYS} days")
    logger.info(f"API Token: {'✅ Found' if ECOURTS_API_TOKEN else '❌ Missing'}")
    logger.info("")
    
    try:
        # 1. Fetch and cache states
        logger.info("1️⃣  Fetching states...")
        states = fetch_states()
        save_cache(states, "states")
        
        # 2. Fetch and cache districts (skip if states failed)
        if not states:
            logger.error("⚠️  No states fetched, skipping districts")
        else:
            logger.info("\n2️⃣  Fetching districts...")
            districts = fetch_all_districts()
            save_cache(districts, "districts")
        
        # 3. Fetch and cache courts (optional - very slow!)
        if include_courts:
            logger.info("\n3️⃣  Fetching courts (this will take a while)...")
            courts = fetch_all_courts()
            save_cache(courts, "courts")
        else:
            logger.info("\n3️⃣  Skipping courts data (use --include-courts to fetch)")
        
        # 4. Fetch and cache consumer forum districts
        logger.info("\n4️⃣  Fetching consumer forum districts...")
        consumer_districts = fetch_consumer_forum_districts()
        save_cache(consumer_districts, "consumer_districts")
        
        # 5. Fetch and cache consumer forum benches
        logger.info("\n5️⃣  Fetching consumer forum benches...")
        consumer_benches = fetch_consumer_forum_benches()
        save_cache(consumer_benches, "consumer_benches")
        
        # 6. Update metadata
        logger.info("\n6️⃣  Updating cache metadata...")
        update_metadata()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ CACHE UPDATE COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Cache location: {CACHE_DIR}")
        logger.info(f"Cache expires: {(datetime.now() + timedelta(days=CACHE_TTL_DAYS)).strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")
        
    except Exception as e:
        logger.error(f"\n❌ Error during cache update: {e}")
        raise


def load_from_cache(cache_key: str) -> Optional[Dict]:
    """Load data from cache file"""
    cache_file = CACHE_FILES[cache_key]
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading cache {cache_key}: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    force = "--force" in sys.argv
    include_courts = "--include-courts" in sys.argv
    
    cache_all_data(force=force, include_courts=include_courts)
