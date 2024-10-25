from datetime import datetime

def format_datetime(dt: datetime) -> str:
    """
    Format datetime to RFC3339 format without microseconds
    
    Args:
        dt (datetime): Datetime object to format
        
    Returns:
        str: RFC3339 formatted date string
    """
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')