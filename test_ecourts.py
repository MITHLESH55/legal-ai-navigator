#!/usr/bin/env python3
"""
Test script for e-Courts API (Kleopatra) integration
"""

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def test_ecourts_api():
    """Test the Kleopatra e-Courts API integration"""
    
    print("=" * 60)
    print("Testing e-Courts API (Kleopatra) Integration")
    print("=" * 60)
    
    api_token = os.getenv("ECOURTS_API_TOKEN")
    
    if not api_token:
        print("❌ ERROR: ECOURTS_API_TOKEN not found in .env file")
        return False
    
    print(f"\n✅ API Token found: {api_token[:20]}...")
    
    # Test 1: Fetch States
    print("\n" + "-" * 60)
    print("Test 1: Fetching available states...")
    print("-" * 60)
    
    try:
        url = "https://court-api.kleopatra.io/api/core/static/district-court/states"
        headers = {"authorization": f"Bearer {api_token}"}
        
        response = requests.get(url, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            states = response.json()
            print(f"✅ SUCCESS: Retrieved {len(states) if isinstance(states, list) else 'N/A'} states")
            print("\nSample Response:")
            print(json.dumps(states[:3] if isinstance(states, list) else states, indent=2))
        else:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False
    
    # Test 2: Test the tool function
    print("\n" + "-" * 60)
    print("Test 2: Testing get_ecourts_states tool...")
    print("-" * 60)
    
    try:
        from tools import get_ecourts_states
        
        result = get_ecourts_states.invoke({})
        print("Tool Result:")
        print(result[:500] + "..." if len(result) > 500 else result)
        print("✅ Tool function works!")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All e-Courts API tests passed!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Try fetching case details with a valid CNR number")
    print("2. Explore other Kleopatra API endpoints")
    print("3. Use the tool in your chat queries")
    print("\nExample query:")
    print('  "Get me the list of states for e-Courts"')
    print('  "Fetch case details for CNR: [your-cnr-number]"')
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_ecourts_api()
