"""
read_mode module
"""
import hashlib
from urllib.parse import urlparse, quote

def transform_link(original_link):
    """
    Transforms the given link into a specific format used by the chrome-distiller.
    
    Args:
    original_link (str): The original URL to be transformed.
    
    Returns:
    str: The transformed URL in the chrome-distiller format.
    """
    parsed_original_link = urlparse(original_link)
    base_part = 'chrome-distiller://00000000-0000-0000-0000-000000000000_'
    hash_value = hashlib.sha256(original_link.encode()).hexdigest()
    final_link = f"{base_part}{hash_value}/?url={quote(original_link)}"
    return final_link