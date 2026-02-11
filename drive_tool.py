import os
import logging
import json
import pickle
import io
from pathlib import Path
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from langchain.tools import tool
try:
    # Try pypdf (newer version)
    from pypdf import PdfReader
except ImportError:
    # Fallback to PyPDF2 (older version)
    from PyPDF2 import PdfReader
import tempfile

# Import data stores for automatic indexing
from data_stores import vector_store, load_and_split_pdf
import graph_builder

# Allow insecure transport for local development (localhost only)
# This is needed because OAuth2 requires HTTPS, but we're running on localhost
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OAuth2 scopes for Google Drive read-only access
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
TOKEN_PICKLE_PATH = Path('./token.pickle')


def _get_drive_service():
    """
    Helper function to get an authenticated Google Drive service.
    Returns the service object if credentials are valid, None otherwise.
    """
    creds = None
    
    # Load existing token if available
    if TOKEN_PICKLE_PATH.exists():
        try:
            with open(TOKEN_PICKLE_PATH, 'rb') as token:
                creds = pickle.load(token)
                logger.info("Loaded existing Google Drive credentials from token.pickle")
        except Exception as e:
            logger.error(f"Error loading token.pickle: {e}")
            return None
    
    # Check if credentials are valid or need refresh
    if creds:
        if creds.expired and creds.refresh_token:
            try:
                logger.info("Refreshing expired Google Drive credentials...")
                creds.refresh(Request())
                # Save refreshed credentials
                with open(TOKEN_PICKLE_PATH, 'wb') as token:
                    pickle.dump(creds, token)
                logger.info("Credentials refreshed and saved")
            except Exception as e:
                logger.error(f"Failed to refresh credentials: {e}")
                return None
        elif not creds.valid:
            logger.warning("Credentials are invalid and cannot be refreshed")
            return None
    else:
        # No credentials available
        logger.info("No valid credentials found")
        return None
    
    # Build and return the Drive service
    try:
        service = build('drive', 'v3', credentials=creds)
        logger.info("Successfully built Google Drive service")
        return service
    except Exception as e:
        logger.error(f"Error building Drive service: {e}")
        return None


@tool
def search_google_drive_files(auto_sync: bool = False) -> str:
    """
    Searches PDF files within the specific '/legal-navigator' Google Drive folder.
    Returns file information including names, IDs, sizes, and dates.
    
    **IMPORTANT**: This tool requires user approval before calling. Once approved,
    it will download and index NEW files into Qdrant (vector search) and Neo4j (knowledge graph)
    ONLY if auto_sync=True.
    
    **DEFAULT BEHAVIOR**: auto_sync=False (only lists files, does NOT download/process).
    **AFTER USER APPROVAL**: The agent should call this again with auto_sync=True to process files.
    
    Use this when the user asks to 'fetch from google drive', 'check drive', 
    'sync drive files', or 'get documents from drive'.
    
    Args:
        auto_sync: If True, downloads and processes NEW files into Qdrant and Neo4j.
                   If False (default), only lists available files without processing.
                   Set to True ONLY after user has approved the Drive tool call.
    """
    logger.info(f"Tool: search_google_drive_files (auto_sync={auto_sync})")
    
    # Get authenticated Drive service
    service = _get_drive_service()
    
    if service is None:
        return json.dumps({
            "status": "Authentication Required",
            "message": "Google Drive access not authorized. Please visit /auth/google to connect your account."
        })
    
    # Get folder ID from environment
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if not folder_id:
        return json.dumps({
            "status": "Configuration Error",
            "message": "GOOGLE_DRIVE_FOLDER_ID not set in environment variables."
        })
    
    try:
        # Search for PDF files in the specified folder
        query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        
        logger.info(f"Searching Google Drive folder {folder_id} for PDF files...")
        results = service.files().list(
            q=query,
            fields="files(id, name, createdTime, modifiedTime, size)",
            pageSize=100
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            logger.info("No PDF files found in the Drive folder")
            return json.dumps({
                "status": "Success",
                "message": "No PDF files found in the '/legal-navigator' folder.",
                "files": []
            })
        
        logger.info(f"Found {len(files)} PDF files in Drive folder")
        
        # Auto-sync: Download and process files into Qdrant and Neo4j
        processed_files = []
        skipped_files = []
        
        if auto_sync:
            logger.info("Auto-sync enabled: Processing files into Qdrant and Neo4j...")
            max_pages = int(os.getenv("MAX_DOCUMENT_PAGES", "20"))
            
            for file in files:
                file_name = file['name']
                file_id = file['id']
                
                try:
                    logger.info(f"Auto-syncing: {file_name}")
                    
                    # Download file
                    request_obj = service.files().get_media(fileId=file_id)
                    file_buffer = io.BytesIO()
                    downloader = MediaIoBaseDownload(file_buffer, request_obj)
                    
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                    
                    file_buffer.seek(0)
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_f:
                        temp_f.write(file_buffer.read())
                        temp_path = temp_f.name
                    
                    try:
                        # Load and split PDF
                        documents = load_and_split_pdf(temp_path)
                        
                        if not documents:
                            logger.warning(f"Failed to load/split {file_name}")
                            skipped_files.append(file_name)
                            continue
                        
                        # Generate case_id from filename
                        case_id = f"drive_{file_name.replace('.pdf', '').replace(' ', '_')}"
                        
                        # Index in Qdrant vector store
                        vector_store.index_documents(documents, case_id, file_name)
                        logger.info(f"✅ Indexed {file_name} in Qdrant")
                        
                        # Check page count for graph building
                        num_pages = len(documents)
                        
                        if num_pages <= max_pages:
                            # Build knowledge graph in Neo4j
                            graph_builder.extract_and_store_graph(documents, case_id, file_name)
                            logger.info(f"✅ Built knowledge graph for {file_name} in Neo4j")
                            processed_files.append(f"{file_name} (vector + graph)")
                        else:
                            logger.info(f"⚠️ Skipped graph for {file_name} ({num_pages} pages > {max_pages} limit)")
                            processed_files.append(f"{file_name} (vector only)")
                    
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                
                except Exception as e:
                    logger.error(f"Error processing {file_name}: {e}")
                    skipped_files.append(file_name)
        
        # Format results
        file_list = [
            {
                "name": file['name'],
                "id": file['id'],
                "created": file.get('createdTime', 'Unknown'),
                "modified": file.get('modifiedTime', 'Unknown'),
                "size": file.get('size', 'Unknown')
            }
            for file in files
        ]
        
        response_data = {
            "status": "Success",
            "message": f"Found {len(files)} PDF file(s) in '/legal-navigator' folder.",
            "files": file_list
        }
        
        if auto_sync:
            response_data["auto_sync"] = {
                "enabled": True,
                "processed": processed_files,
                "skipped": skipped_files,
                "summary": f"Processed {len(processed_files)} files into Qdrant/Neo4j. Skipped: {len(skipped_files)}."
            }
        
        return json.dumps(response_data, indent=2)
        
    except HttpError as error:
        logger.error(f"Google Drive API error: {error}")
        return json.dumps({
            "status": "API Error",
            "message": f"Failed to search Google Drive: {str(error)}"
        })
    except Exception as e:
        logger.error(f"Unexpected error searching Drive: {e}")
        return json.dumps({
            "status": "Error",
            "message": f"Unexpected error: {str(e)}"
        })


# Optional: Helper functions for future PDF reading capability
# Keeping these for when we need to actually download and extract PDF content
def _download_and_extract_text(file_id: str, file_name: str) -> str:
    """
    Helper function to download a PDF file from Google Drive and extract its text content.
    Returns the extracted text or an error message.
    NOTE: Currently not exposed as a tool - keeping for future use.
    """
    logger.info(f"Downloading and extracting text from file: {file_name} (ID: {file_id})")
    
    service = _get_drive_service()
    if service is None:
        return "Error: Google Drive authentication failed"
    
    try:
        # Download the file
        request = service.files().get_media(fileId=file_id)
        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
            logger.info(f"Download progress: {int(status.progress() * 100)}%")
        
        # Reset stream position
        file_stream.seek(0)
        
        # Extract text from PDF
        try:
            pdf_reader = PdfReader(file_stream)
            text_content = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
            
            full_text = "\n\n".join(text_content)
            logger.info(f"Successfully extracted {len(full_text)} characters from {file_name}")
            
            return full_text if full_text else "Error: No text content could be extracted from PDF"
            
        except Exception as pdf_error:
            logger.error(f"Error extracting text from PDF: {pdf_error}")
            return f"Error: Failed to extract text from PDF - {str(pdf_error)}"
            
    except HttpError as error:
        logger.error(f"Google Drive API error during download: {error}")
        return f"Error: Failed to download file from Google Drive - {str(error)}"
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        return f"Error: Unexpected error - {str(e)}"
