"""
Audixa SDK Utility Functions.

Provides logging, retry logic, file operations, and helper utilities.
"""

from __future__ import annotations

import logging
import os
import random
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

import aiohttp
import requests

from .config import (
    FORMAT_EXTENSIONS,
    SUPPORTED_FORMATS,
    AudioFormat,
    is_format_supported,
)
from .exceptions import (
    AudixaError,
    NetworkError,
    RateLimitError,
    TimeoutError,
    UnsupportedFormatError,
)

# =============================================================================
# Logging Setup
# =============================================================================

logger = logging.getLogger("audixa")
"""
SDK logger instance.

Enable debug logging with:
    >>> import logging
    >>> logging.basicConfig(level=logging.DEBUG)
    
Or configure the audixa logger specifically:
    >>> logging.getLogger("audixa").setLevel(logging.DEBUG)
"""


# =============================================================================
# Type Variables for Generics
# =============================================================================

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Retry Logic
# =============================================================================

def calculate_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> float:
    """
    Calculate exponential backoff delay with optional jitter.
    
    Args:
        attempt: The current attempt number (0-indexed).
        base_delay: Base delay in seconds.
        max_delay: Maximum delay cap in seconds.
        jitter: Whether to add random jitter.
        
    Returns:
        Delay in seconds before the next retry.
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    if jitter:
        delay = delay * (0.5 + random.random())  # 50-100% of calculated delay
    return delay


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error should trigger a retry.
    
    Retryable errors include:
    - Network errors (connection, timeout)
    - Rate limit errors (429)
    - Server errors (5xx)
    
    Args:
        error: The exception to check.
        
    Returns:
        True if the error is retryable, False otherwise.
    """
    if isinstance(error, RateLimitError):
        return True
    if isinstance(error, NetworkError):
        return True
    if isinstance(error, AudixaError) and error.status_code:
        return error.status_code >= 500 or error.status_code == 429
    return False


def retry_sync(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Callable[[F], F]:
    """
    Decorator for synchronous functions with retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries in seconds.
        max_delay: Maximum delay cap in seconds.
        
    Returns:
        Decorated function with retry logic.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if not is_retryable_error(e) or attempt >= max_retries:
                        raise
                    delay = calculate_backoff(attempt, base_delay, max_delay)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
            raise last_error  # type: ignore
        return wrapper  # type: ignore
    return decorator


async def retry_async_operation(
    operation: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Any:
    """
    Execute an async operation with retry logic.
    
    Args:
        operation: Async callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries in seconds.
        max_delay: Maximum delay cap in seconds.
        
    Returns:
        Result of the operation.
        
    Raises:
        The last exception if all retries fail.
    """
    import asyncio
    
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except Exception as e:
            last_error = e
            if not is_retryable_error(e) or attempt >= max_retries:
                raise
            delay = calculate_backoff(attempt, base_delay, max_delay)
            logger.warning(
                f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                f"Retrying in {delay:.2f}s..."
            )
            await asyncio.sleep(delay)
    raise last_error  # type: ignore


# =============================================================================
# File Operations
# =============================================================================

def validate_output_filepath(filepath: str | Path) -> Path:
    """
    Validate and normalize output file path.
    
    Checks that:
    - The file extension is supported
    - The parent directory exists or can be created
    
    Args:
        filepath: The output file path.
        
    Returns:
        Normalized Path object.
        
    Raises:
        UnsupportedFormatError: If the file extension is not supported.
        OSError: If the parent directory cannot be created.
    """
    path = Path(filepath)
    extension = path.suffix.lower()
    
    # Check if extension is supported
    if extension:
        format_name = extension.lstrip(".")
        if not is_format_supported(format_name):
            raise UnsupportedFormatError(
                requested_format=format_name,
                supported_formats=SUPPORTED_FORMATS,
            )
    else:
        # No extension provided, add default
        path = path.with_suffix(FORMAT_EXTENSIONS["wav"])
        logger.debug(f"No extension provided, using default: {path}")
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    return path


def download_file_sync(
    url: str,
    filepath: Path,
    timeout: float = 60.0,
    chunk_size: int = 8192,
) -> Path:
    """
    Download a file synchronously.
    
    Args:
        url: URL to download from.
        filepath: Local path to save the file.
        timeout: Request timeout in seconds.
        chunk_size: Download chunk size in bytes.
        
    Returns:
        Path to the downloaded file.
        
    Raises:
        NetworkError: If the download fails.
    """
    logger.debug(f"Downloading file from {url} to {filepath}")
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"Downloaded file to {filepath}")
        return filepath
        
    except requests.exceptions.Timeout as e:
        raise TimeoutError(f"Download timed out after {timeout}s") from e
    except requests.exceptions.RequestException as e:
        raise NetworkError(f"Failed to download file: {e}", original_error=e) from e
    except IOError as e:
        raise NetworkError(f"Failed to write file: {e}", original_error=e) from e


async def download_file_async(
    url: str,
    filepath: Path,
    timeout: float = 60.0,
    chunk_size: int = 8192,
) -> Path:
    """
    Download a file asynchronously.
    
    Args:
        url: URL to download from.
        filepath: Local path to save the file.
        timeout: Request timeout in seconds.
        chunk_size: Download chunk size in bytes.
        
    Returns:
        Path to the downloaded file.
        
    Raises:
        NetworkError: If the download fails.
    """
    import aiofiles
    
    logger.debug(f"Downloading file from {url} to {filepath}")
    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                
                async with aiofiles.open(filepath, "wb") as f:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        await f.write(chunk)
        
        logger.info(f"Downloaded file to {filepath}")
        return filepath
        
    except aiohttp.ClientError as e:
        raise NetworkError(f"Failed to download file: {e}", original_error=e) from e
    except IOError as e:
        raise NetworkError(f"Failed to write file: {e}", original_error=e) from e


# =============================================================================
# API Helpers
# =============================================================================

def build_url(base_url: str, endpoint: str, **path_params: str) -> str:
    """
    Build a full API URL from base URL and endpoint.
    
    Args:
        base_url: API base URL.
        endpoint: Endpoint path (may contain {placeholders}).
        **path_params: Values to substitute into placeholders.
        
    Returns:
        Complete URL string.
    """
    formatted_endpoint = endpoint.format(**path_params) if path_params else endpoint
    return f"{base_url.rstrip('/')}{formatted_endpoint}"


def validate_text(text: str) -> str:
    """
    Validate text input for TTS.
    
    Args:
        text: The text to validate.
        
    Returns:
        The validated text (stripped of leading/trailing whitespace).
        
    Raises:
        ValidationError: If the text is invalid.
    """
    from .exceptions import ValidationError
    
    if not text:
        raise ValidationError("Text cannot be empty.")
    
    text = text.strip()
    if not text:
        raise ValidationError("Text cannot be empty or contain only whitespace.")
    
    return text
