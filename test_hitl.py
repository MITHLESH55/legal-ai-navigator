"""
Enhanced Test Client for VoiceLegal Backend with HITL Support
Tests human-in-the-loop approval flow for sensitive tools.
"""
import requests
import json
import time

BASE_URL = "http://localhost:7860"

headers = {
    "Content-Type": "application/json"
}

def test_chat_with_approval():
    """Test chat endpoint with approval flow."""
    print("\n" + "="*70)
    print("TEST: Chat with Tool Approval (Google Drive)")
    print("="*70)
    
    # Query that triggers approval
    query = "Search my Google Drive for legal documents"
    print(f"\n📤 Sending query: '{query}'")
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"query": query},
        headers=headers,
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        
        # Check if approval needed
        if result.get("needs_approval"):
            print("\n⚠️  APPROVAL REQUIRED")
            print(f"Session ID: {result['session_id']}")
            print(f"\nMessage: {result.get('response', '')}")
            print("\nTools requesting approval:")
            
            for req in result.get("approval_requests", []):
                print(f"\n  🔧 Tool: {req['tool_name']}")
                print(f"  📝 Request: {req['approval_message']}")
            
            # Get user decision
            user_input = input("\n➡️  Approve tool execution? (yes/no): ").strip().lower()
            approved = user_input in ['yes', 'y']
            
            print(f"\n📤 Sending approval: {approved}")
            
            # Send approval response
            approval_response = requests.post(
                f"{BASE_URL}/chat/approve",
                json={
                    "session_id": result["session_id"],
                    "approved": approved
                },
                headers=headers,
                timeout=60
            )
            
            if approval_response.status_code == 200:
                final_result = approval_response.json()
                print("\n✅ SUCCESS")
                print(f"Response: {final_result.get('response', '')}")
            else:
                print(f"\n❌ ERROR: {approval_response.status_code}")
                print(approval_response.text)
        else:
            print("\n✅ SUCCESS (No approval needed)")
            print(f"Response: {result.get('response', '')}")
    else:
        print(f"\n❌ ERROR: {response.status_code}")
        print(response.text)


def test_chat_no_approval():
    """Test chat endpoint without approval (direct answer)."""
    print("\n" + "="*70)
    print("TEST: Chat without Approval (Direct Answer)")
    print("="*70)
    
    query = "What is IPC Section 302?"
    print(f"\n📤 Sending query: '{query}'")
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"query": query},
        headers=headers,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ SUCCESS")
        print(f"Response: {result.get('response', '')[:500]}...")
        
        if result.get("needs_approval"):
            print("\n⚠️  Unexpected: Approval requested for general question")
    else:
        print(f"\n❌ ERROR: {response.status_code}")
        print(response.text)


def test_browser_search_approval():
    """Test browser search approval flow."""
    print("\n" + "="*70)
    print("TEST: Browser Search with Approval")
    print("="*70)
    
    query = "Open a browser search for Supreme Court judgments on IPC 302"
    print(f"\n📤 Sending query: '{query}'")
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"query": query},
        headers=headers,
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        
        if result.get("needs_approval"):
            print("\n⚠️  APPROVAL REQUIRED")
            print(f"Session ID: {result['session_id']}")
            print(f"\nMessage: {result.get('response', '')}")
            
            for req in result.get("approval_requests", []):
                print(f"\n  🔧 Tool: {req['tool_name']}")
                print(f"  📝 Request: {req['approval_message']}")
            
            user_input = input("\n➡️  Approve tool execution? (yes/no): ").strip().lower()
            approved = user_input in ['yes', 'y']
            
            print(f"\n📤 Sending approval: {approved}")
            
            approval_response = requests.post(
                f"{BASE_URL}/chat/approve",
                json={
                    "session_id": result["session_id"],
                    "approved": approved
                },
                headers=headers,
                timeout=60
            )
            
            if approval_response.status_code == 200:
                final_result = approval_response.json()
                print("\n✅ SUCCESS")
                
                # Check for browser redirect
                if final_result.get("action") == "BROWSER_REDIRECT":
                    print(f"🌐 Browser redirect: {final_result.get('url')}")
                else:
                    print(f"Response: {final_result.get('response', '')}")
            else:
                print(f"\n❌ ERROR: {approval_response.status_code}")
                print(approval_response.text)
        else:
            print("\n⚠️  Expected approval but got direct response")
            print(f"Response: {result.get('response', '')}")
    else:
        print(f"\n❌ ERROR: {response.status_code}")
        print(response.text)


def test_uploaded_docs_no_approval():
    """Test that uploaded docs search doesn't require approval."""
    print("\n" + "="*70)
    print("TEST: Uploaded Documents Search (Should Not Need Approval)")
    print("="*70)
    
    query = "Summarize my uploaded documents"
    print(f"\n📤 Sending query: '{query}'")
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"query": query},
        headers=headers,
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        
        if result.get("needs_approval"):
            print("\n⚠️  WARNING: Unexpected approval request")
            print(f"This tool should not require approval")
            print(f"Tools: {result.get('approval_requests', [])}")
        else:
            print("\n✅ SUCCESS (No approval needed, as expected)")
            print(f"Response: {result.get('response', '')[:300]}...")
    else:
        print(f"\n❌ ERROR: {response.status_code}")
        print(response.text)


def test_rejection_flow():
    """Test what happens when user rejects tool approval."""
    print("\n" + "="*70)
    print("TEST: Tool Approval Rejection")
    print("="*70)
    
    query = "Search my Google Drive for contracts"
    print(f"\n📤 Sending query: '{query}'")
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"query": query},
        headers=headers,
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        
        if result.get("needs_approval"):
            print("\n⚠️  APPROVAL REQUIRED")
            print(f"Session ID: {result['session_id']}")
            
            print(f"\n❌ Rejecting approval (simulated)")
            
            # Reject the tool
            approval_response = requests.post(
                f"{BASE_URL}/chat/approve",
                json={
                    "session_id": result["session_id"],
                    "approved": False  # Reject
                },
                headers=headers,
                timeout=60
            )
            
            if approval_response.status_code == 200:
                final_result = approval_response.json()
                print("\n✅ Rejection handled gracefully")
                print(f"Response: {final_result.get('response', '')}")
            else:
                print(f"\n❌ ERROR: {approval_response.status_code}")
                print(approval_response.text)
        else:
            print("\n⚠️  Expected approval but got direct response")
    else:
        print(f"\n❌ ERROR: {response.status_code}")
        print(response.text)


def run_all_hitl_tests():
    """Run all HITL tests."""
    print("\n" + "="*70)
    print("🚀 HUMAN-IN-THE-LOOP TEST SUITE")
    print("="*70)
    print("\nThis will test the approval flow for sensitive tools.")
    print("You'll be prompted to approve/reject actions.\n")
    
    input("Press Enter to start tests...")
    
    try:
        # Test 1: Direct answer (no approval)
        test_chat_no_approval()
        time.sleep(2)
        
        # Test 2: Uploaded docs (no approval)
        test_uploaded_docs_no_approval()
        time.sleep(2)
        
        # Test 3: Google Drive approval flow
        test_chat_with_approval()
        time.sleep(2)
        
        # Test 4: Browser search approval flow
        test_browser_search_approval()
        time.sleep(2)
        
        # Test 5: Rejection flow
        test_rejection_flow()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")


def interactive_menu():
    """Interactive menu for testing."""
    while True:
        print("\n" + "="*70)
        print("🧪 VoiceLegal HITL Test Menu")
        print("="*70)
        print("\n1. Test Direct Answer (No Approval)")
        print("2. Test Uploaded Docs Search (No Approval)")
        print("3. Test Google Drive Search (With Approval)")
        print("4. Test Browser Search (With Approval)")
        print("5. Test Approval Rejection")
        print("6. Run All HITL Tests")
        print("0. Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            test_chat_no_approval()
        elif choice == "2":
            test_uploaded_docs_no_approval()
        elif choice == "3":
            test_chat_with_approval()
        elif choice == "4":
            test_browser_search_approval()
        elif choice == "5":
            test_rejection_flow()
        elif choice == "6":
            run_all_hitl_tests()
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
        else:
            print("\n❌ Invalid option")


if __name__ == "__main__":
    print("="*70)
    print("VoiceLegal Backend - Human-in-the-Loop Test Client")
    print("="*70)
    print(f"\nBackend URL: {BASE_URL}")
    print("\nThis client tests the approval flow for sensitive tools:")
    print("  • Google Drive search")
    print("  • Browser search")
    print("\nMake sure the backend is running before starting tests.\n")
    
    interactive_menu()
