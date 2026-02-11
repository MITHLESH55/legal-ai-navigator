import requests
import os
import json
from pathlib import Path
import time # Import time for potential delays

# --- Configuration ---
BASE_URL = "http://localhost:7860"  # Your Flask app's address
UPLOAD_ENDPOINT = f"{BASE_URL}/upload"
CHAT_ENDPOINT = f"{BASE_URL}/chat"
AUTH_STATUS_ENDPOINT = f"{BASE_URL}/auth/status"
SYNC_DRIVE_ENDPOINT = f"{BASE_URL}/sync_drive"
CASES_ENDPOINT = f"{BASE_URL}/cases"

# --- Helper Functions ---

def print_response(response: requests.Response):
    """Prints the status code and JSON response nicely."""
    print(f"\n--- Response ---")
    print(f"Status Code: {response.status_code}")
    try:
        print("JSON Body:")
        print(json.dumps(response.json(), indent=2))
    except json.JSONDecodeError:
        print("Body (Not JSON):")
        print(response.text)
    print("----------------\n")

# --- Test Functions ---

def upload_pdf_and_verify_processing():
    """Prompts for PDF path, uploads it, and asks user to check logs."""
    pdf_path_str = input("Enter the path to the PDF file for upload test: ")
    pdf_path = Path(pdf_path_str)

    if not pdf_path.is_file() or pdf_path.suffix.lower() != '.pdf':
        print("Invalid file path or not a PDF. Please try again.")
        return

    case_id = input("Enter a case_id (optional, press Enter to auto-generate): ")
    payload = {}
    if case_id:
        payload['case_id'] = case_id

    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': (pdf_path.name, f, 'application/pdf')}
            print(f"\nUploading {pdf_path.name}...")
            response = requests.post(UPLOAD_ENDPOINT, files=files, data=payload)
            print_response(response)

            if response.ok:
                print("\n--- Verification ---")
                print("Upload accepted. Please check your **BACKEND LOGS** now.")
                print("Look for messages indicating:")
                print("  - 'Starting graph extraction...'")
                print("  - 'Using LLM batching...' (for <= 10 pages)")
                print("  - 'Using NER model...' (for 11-20 pages)")
                print("  - 'Graph building skipped (document exceeds page limit)' (for > 20 pages)")
                print("  - 'Stored graph for chunk...' or 'Stored graph for full_document...'")
                print("  - '[Background Thread] Finished graph extraction...'")
                input("Press Enter after checking the logs...")
                # You might want to store the returned case_id
                try:
                    returned_case_id = response.json().get("case_id")
                    if returned_case_id:
                         print(f"Associated case_id: {returned_case_id}")
                except Exception:
                    pass
            else:
                 print("Upload failed or was rejected by the backend.")


    except FileNotFoundError:
        print(f"Error: File not found at {pdf_path}")
    except Exception as e:
        print(f"An error occurred during upload: {e}")


def upload_and_chat_session():
    """
    Enhanced workflow: Upload a PDF first, then start continuous chat session.
    This allows you to ask questions about the uploaded document.
    """
    print("\n--- Upload & Chat Workflow ---")
    
    # Step 1: Upload PDF
    pdf_path_str = input("Enter the path to the PDF file to upload (or press Enter to skip): ")
    
    case_id = None
    session_id = None
    
    if pdf_path_str.strip():
        pdf_path = Path(pdf_path_str.strip())
        
        if not pdf_path.is_file() or pdf_path.suffix.lower() != '.pdf':
            print("Invalid file path or not a PDF. Skipping upload.")
        else:
            # Upload the PDF
            case_id_input = input("Enter a case_id (optional, press Enter to auto-generate): ")
            payload = {}
            if case_id_input:
                payload['case_id'] = case_id_input
                case_id = case_id_input
            
            try:
                with open(pdf_path, 'rb') as f:
                    files = {'file': (pdf_path.name, f, 'application/pdf')}
                    print(f"\n📤 Uploading {pdf_path.name}...")
                    response = requests.post(UPLOAD_ENDPOINT, files=files, data=payload)
                    
                    if response.ok:
                        result = response.json()
                        case_id = result.get("case_id", case_id)
                        print(f"✅ Upload successful!")
                        print(f"📋 Case ID: {case_id}")
                        print(f"📄 File: {result.get('file_name', pdf_path.name)}")
                        print(f"📊 Status: {result.get('status', 'Processing...')}")
                        print("\n💡 Tip: Check backend logs for processing details")
                        print("    (graph extraction, vector indexing, etc.)\n")
                    else:
                        print(f"❌ Upload failed: {response.status_code}")
                        try:
                            print(response.json())
                        except:
                            print(response.text)
                        case_id = None
                        
            except FileNotFoundError:
                print(f"Error: File not found at {pdf_path}")
            except Exception as e:
                print(f"An error occurred during upload: {e}")
    
    # Step 2: Start continuous chat
    print("\n--- Continuous Chat Session ---")
    print("💬 You can now ask questions about your uploaded document or anything else.")
    print("📌 Type 'quit' to exit.\n")
    
    if case_id:
        print(f"🔗 Associated with Case ID: {case_id}")
    
    while True:
        query = input("\n👤 You: ")
        if query.lower() == 'quit':
            print("\n👋 Goodbye!")
            break
        
        if not query.strip():
            continue
        
        payload = {
            "query": query,
            "case_id": case_id if case_id else None
        }
        if session_id:
            payload["session_id"] = session_id
        
        try:
            response = requests.post(CHAT_ENDPOINT, json=payload)
            
            if response.ok:
                data = response.json()
                
                # Update session_id for conversation continuity
                if "session_id" in data and not session_id:
                    session_id = data["session_id"]
                
                # Handle HITL approval request
                if data.get("needs_approval"):
                    print("\n⚠️  APPROVAL REQUIRED")
                    print(f"Message: {data.get('response', '')}")
                    print("\nThe following actions require your approval:")
                    
                    for req in data.get("approval_requests", []):
                        print(f"\n  🔧 Tool: {req.get('tool_name', 'Unknown')}")
                        print(f"  📝 {req.get('approval_message', 'Approve this action?')}")
                    
                    # Get user decision
                    user_input = input("\n➡️  Approve? (yes/no): ").strip().lower()
                    approved = user_input in ['yes', 'y']
                    
                    # Send approval response
                    approval_payload = {
                        "session_id": session_id,
                        "approved": approved
                    }
                    
                    print(f"\n{'✅' if approved else '❌'} Sending {'approval' if approved else 'rejection'}...")
                    
                    approval_response = requests.post(
                        f"{BASE_URL}/chat/approve",
                        json=approval_payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if approval_response.ok:
                        approval_data = approval_response.json()
                        print("\n🤖 Assistant:")
                        
                        # Handle browser redirect
                        if approval_data.get("action") == "BROWSER_REDIRECT":
                            print(f"  🌐 (Action Required: Open this URL in your browser)")
                            print(f"  🔗 URL: {approval_data.get('url')}")
                        elif "response" in approval_data:
                            print(f"  {approval_data['response']}")
                        else:
                            print(f"  {json.dumps(approval_data, indent=2)}")
                    else:
                        print(f"\n❌ Error sending approval: {approval_response.status_code}")
                        try:
                            print(approval_response.json())
                        except:
                            print(approval_response.text)
                
                # Handle clarification request
                elif data.get("needs_clarification"):
                    print("\n🤖 Assistant:")
                    print(f"  {data.get('response', 'Please provide more information.')}")
                
                # Handle browser redirect action
                elif data.get("action") == "BROWSER_REDIRECT":
                    print("\n🤖 Assistant:")
                    print(f"  🌐 (Action Required: Open this URL in your browser)")
                    print(f"  🔗 URL: {data.get('url')}")
                
                # Handle normal text response
                elif "response" in data:
                    print("\n🤖 Assistant:")
                    print(f"  {data['response']}")
                else:
                    print("\n🤖 Assistant:")
                    print(f"  {json.dumps(data, indent=2)}")
            
            else:
                print(f"\n❌ Error: {response.status_code}")
                try:
                    print(response.json())
                except Exception:
                    print(response.text)
        
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Could not connect to the chat endpoint: {e}")
        except Exception as e:
            print(f"\n❌ An error occurred during chat: {e}")
    
    print("\n--- End Chat Session ---\n")


def chat_session():
    """Legacy chat session (kept for backward compatibility)."""
    print("\n⚠️  Tip: Use 'Upload & Chat Workflow' (option 2) for better experience!")
    print("--- Start Chat Session ---")
    print("Type 'quit' to exit.")
    session_id = None
    case_id = input("Enter case_id to associate with this chat (optional, press Enter for none): ")

    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'quit':
            break

        payload = {
            "query": query,
            "case_id": case_id if case_id else None
        }
        if session_id:
            payload["session_id"] = session_id

        try:
            response = requests.post(CHAT_ENDPOINT, json=payload)

            if response.ok:
                data = response.json()
                
                # Update session_id for next turn
                if "session_id" in data and not session_id:
                    session_id = data["session_id"]
                
                # Check if approval is needed (HITL)
                if data.get("needs_approval"):
                    print("\n⚠️  APPROVAL REQUIRED")
                    print(f"Message: {data.get('response', '')}")
                    print("\nThe following actions require your approval:")
                    
                    for req in data.get("approval_requests", []):
                        print(f"\n  🔧 Tool: {req.get('tool_name', 'Unknown')}")
                        print(f"  📝 {req.get('approval_message', 'Approve this action?')}")
                    
                    # Get user decision
                    user_input = input("\n➡️  Approve? (yes/no): ").strip().lower()
                    approved = user_input in ['yes', 'y']
                    
                    # Send approval response
                    approval_payload = {
                        "session_id": session_id,
                        "approved": approved
                    }
                    
                    print(f"\n{'✅' if approved else '❌'} Sending {'approval' if approved else 'rejection'}...")
                    
                    approval_response = requests.post(
                        f"{BASE_URL}/chat/approve",
                        json=approval_payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if approval_response.ok:
                        approval_data = approval_response.json()
                        print("\nAssistant:")
                        
                        # Handle browser redirect
                        if approval_data.get("action") == "BROWSER_REDIRECT":
                            print(f"(Action Required: Open this URL in your browser)")
                            print(f"  URL: {approval_data.get('url')}")
                        elif "response" in approval_data:
                            print(approval_data["response"])
                        else:
                            print(json.dumps(approval_data, indent=2))
                    else:
                        print(f"\n❌ Error sending approval: {approval_response.status_code}")
                        try:
                            print(approval_response.json())
                        except:
                            print(approval_response.text)
                
                # Check for clarification request
                elif data.get("needs_clarification"):
                    print("\nAssistant:")
                    print(data.get("response", "Please provide more information."))
                
                # Handle browser redirect action
                elif data.get("action") == "BROWSER_REDIRECT":
                    print("\nAssistant:")
                    print(f"(Action Required: Open this URL in your browser)")
                    print(f"  URL: {data.get('url')}")
                
                # Handle normal text response
                elif "response" in data:
                    print("\nAssistant:")
                    print(data["response"])
                else:
                    print("\nAssistant:")
                    print(json.dumps(data, indent=2)) # Print raw if 'response' is missing

            else:
                 print(f"Error: {response.status_code}")
                 try:
                     print(response.json())
                 except Exception:
                     print(response.text)

        except requests.exceptions.RequestException as e:
            print(f"Could not connect to the chat endpoint: {e}")
        except Exception as e:
            print(f"An error occurred during chat: {e}")

    print("--- End Chat Session ---\n")


def check_auth_status():
    """Checks the Google Drive authentication status."""
    print("\nChecking Google Drive Auth Status...")
    try:
        response = requests.get(AUTH_STATUS_ENDPOINT)
        print_response(response)
        try:
            if not response.json().get("authenticated"):
                 print("\nNote: Backend is not authenticated with Google Drive.")
                 print(f"Visit {BASE_URL}/auth/google in your browser (while backend runs) to authenticate.")
        except Exception:
            pass
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to the auth status endpoint: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def sync_google_drive():
    """Initiates the Google Drive sync process."""
    print("\nInitiating Google Drive Sync...")
    print("(This might take a while depending on the number of files)")
    try:
        # Adding a timeout as sync can be long
        # Send empty JSON object with proper Content-Type header
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            SYNC_DRIVE_ENDPOINT, 
            json={},  # Empty JSON body is fine, endpoint uses it optionally
            headers=headers,
            timeout=300  # 5 minute timeout
        )
        print_response(response)
        if response.status_code == 401:
             print("\nError: Backend not authenticated with Google Drive.")
             print(f"Visit {BASE_URL}/auth/google in your browser (while backend runs) to authenticate first.")

    except requests.exceptions.Timeout:
         print("Error: The sync request timed out ( > 5 minutes). Check backend logs for progress.")
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to the sync endpoint: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def list_cases():
    """Lists processed cases from the backend."""
    print("\nFetching list of processed cases...")
    try:
        response = requests.get(CASES_ENDPOINT)
        print_response(response)
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to the cases endpoint: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


# --- Main Menu ---

def main():
    while True:
        print("\n--- Backend Test Menu ---")
        print("1. Upload PDF & Verify Processing (Check Logs)")
        print("2. 🎯 Upload & Chat Workflow (Recommended)")
        print("3. Chat Session Only")
        print("4. Check Google Drive Auth Status")
        print("5. Sync Files from Google Drive Folder")
        print("6. List Processed Cases")
        print("0. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            upload_pdf_and_verify_processing()
        elif choice == '2':
            upload_and_chat_session()  # New recommended workflow
        elif choice == '3':
            chat_session()  # Legacy chat only
        elif choice == '4':
            check_auth_status()
        elif choice == '5':
            sync_google_drive()
        elif choice == '6':
            list_cases()
        elif choice == '0':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

