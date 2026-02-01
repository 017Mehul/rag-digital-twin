"""
File utilities for the RAG Digital Twin system.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import mimetypes


def ensure_directory(directory_path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        Path object for the directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    return Path(file_path).stat().st_size


def is_valid_file(file_path: str, max_size_mb: Optional[int] = None) -> Tuple[bool, str]:
    """
    Check if a file is valid for processing.
    
    Args:
        file_path: Path to file
        max_size_mb: Maximum file size in MB (optional)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        return False, f"File does not exist: {file_path}"
    
    # Check if it's a file (not directory)
    if not path.is_file():
        return False, f"Path is not a file: {file_path}"
    
    # Check file size
    if max_size_mb:
        file_size_mb = get_file_size(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB"
    
    # Check if file is readable
    if not os.access(file_path, os.R_OK):
        return False, f"File is not readable: {file_path}"
    
    return True, ""


def get_supported_file_types() -> List[str]:
    """
    Get list of supported file types.
    
    Returns:
        List of supported file extensions
    """
    return ['.pdf', '.txt', '.md']


def is_supported_file_type(file_path: str) -> bool:
    """
    Check if a file type is supported.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file type is supported
    """
    extension = Path(file_path).suffix.lower()
    return extension in get_supported_file_types()


def get_file_type(file_path: str) -> str:
    """
    Get the file type based on extension.
    
    Args:
        file_path: Path to file
        
    Returns:
        File type string (pdf, txt, md, unknown)
    """
    extension = Path(file_path).suffix.lower()
    
    type_mapping = {
        '.pdf': 'pdf',
        '.txt': 'txt',
        '.md': 'md',
        '.markdown': 'md'
    }
    
    return type_mapping.get(extension, 'unknown')


def clean_filename(filename: str) -> str:
    """
    Clean a filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Remove invalid characters for most filesystems
    invalid_chars = '<>:"/\\|?*'
    cleaned = filename
    
    for char in invalid_chars:
        cleaned = cleaned.replace(char, '_')
    
    # Remove leading/trailing whitespace and dots
    cleaned = cleaned.strip(' .')
    
    # Ensure filename is not empty
    if not cleaned:
        cleaned = "unnamed_file"
    
    return cleaned


def backup_file(file_path: str, backup_dir: Optional[str] = None) -> str:
    """
    Create a backup copy of a file.
    
    Args:
        file_path: Path to file to backup
        backup_dir: Directory for backup (optional)
        
    Returns:
        Path to backup file
    """
    source_path = Path(file_path)
    
    if backup_dir:
        backup_path = Path(backup_dir)
        ensure_directory(backup_dir)
    else:
        backup_path = source_path.parent
    
    # Create backup filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{source_path.stem}_{timestamp}{source_path.suffix}"
    backup_file_path = backup_path / backup_filename
    
    # Copy file
    shutil.copy2(source_path, backup_file_path)
    
    return str(backup_file_path)


def find_files(directory: str, pattern: str = "*", recursive: bool = True) -> List[str]:
    """
    Find files matching a pattern in a directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern (glob style)
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    path = Path(directory)
    
    if not path.exists():
        return []
    
    if recursive:
        files = path.rglob(pattern)
    else:
        files = path.glob(pattern)
    
    return [str(f) for f in files if f.is_file()]


def get_directory_size(directory: str) -> int:
    """
    Get the total size of all files in a directory.
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    
    for file_path in find_files(directory):
        try:
            total_size += get_file_size(file_path)
        except (OSError, FileNotFoundError):
            # Skip files that can't be accessed
            continue
    
    return total_size


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"