import logging
import json
from langchain.tools import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def open_browser_search(query: str) -> str:
    """
    Signals the frontend to open the AI Browser application at http://localhost:5521
    with a specific search query. The query is URL-encoded and passed as a parameter.
    Use this tool when the user asks to perform a complex web task, research a topic,
    or browse to a specific website. This tool does not return results itself; it
    triggers a redirect on the client-side.
    """
    logger.info(f"Tool: open_browser_search - Query: {query}")
    
    # Format the query for a URL parameter, replacing spaces with '+'
    formatted_query = query.replace(' ', '+')
    
    # Return a JSON signal that the frontend will intercept
    return json.dumps({
        "action": "BROWSER_REDIRECT",
        "url": f"http://localhost:5521/?query={formatted_query}"
    })
