"""
Audixa SDK Configuration Module.

Centralized configuration for API settings, audio formats, and defaults.
Designed for easy extensibility as new features/formats are added.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

# =============================================================================
# Audio Format Configuration (Extensibility Point)
# =============================================================================
# To add new formats in the future:
# 1. Add format string to SUPPORTED_FORMATS tuple
# 2. Update AudioFormat Literal type
# 3. Add corresponding MIME type to FORMAT_MIME_TYPES
# 4. Add file extension to FORMAT_EXTENSIONS

SUPPORTED_FORMATS: tuple[str, ...] = ("wav",)
"""Currently supported audio output formats."""

AudioFormat = Literal["wav"]
"""Type alias for supported audio formats. Update when adding new formats."""

FORMAT_MIME_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    # Future: "mp3": "audio/mpeg", "ogg": "audio/ogg", "flac": "audio/flac"
}
"""MIME types for each supported format."""

FORMAT_EXTENSIONS: dict[str, str] = {
    "wav": ".wav",
    # Future: "mp3": ".mp3", "ogg": ".ogg", "flac": ".flac"
}
"""File extensions for each supported format."""

DEFAULT_FORMAT: AudioFormat = "wav"
"""Default audio output format."""


# =============================================================================
# API Configuration
# =============================================================================

DEFAULT_BASE_URL: str = "https://api.audixa.ai/v2"
"""Default Audixa API base URL."""

DEFAULT_TIMEOUT: float = 30.0
"""Default HTTP request timeout in seconds."""

DEFAULT_MAX_RETRIES: int = 3
"""Default maximum number of retries for failed requests."""

DEFAULT_POLL_INTERVAL: float = 1.0
"""Default polling interval for status checks (seconds)."""

DEFAULT_WAIT_TIMEOUT: float = 360.0
"""Default timeout for tts_and_wait operations (seconds)."""

DEFAULT_MAX_CONCURRENCY: int = 10
"""Default maximum concurrent requests for async client."""


# =============================================================================
# API Endpoints
# =============================================================================

ENDPOINTS: dict[str, str] = {
    "tts": "/tts",
    "status": "/status",
    "voices": "/voices",
}
"""API endpoint paths."""


# =============================================================================
# Global Configuration State
# =============================================================================

@dataclass
class GlobalConfig:
    """
    Global configuration singleton for module-level API functions.
    
    This class stores the global API key and settings used by the 
    convenience functions in the top-level audixa module.
    
    Attributes:
        api_key: The Audixa API key. Falls back to AUDIXA_API_KEY env var.
        base_url: API base URL.
        timeout: Default request timeout.
        max_retries: Maximum retry attempts.
        default_format: Default audio output format.
    """
    api_key: str | None = field(default=None)
    base_url: str = field(default=DEFAULT_BASE_URL)
    timeout: float = field(default=DEFAULT_TIMEOUT)
    max_retries: int = field(default=DEFAULT_MAX_RETRIES)
    default_format: AudioFormat = field(default=DEFAULT_FORMAT)
    poll_interval: float = field(default=DEFAULT_POLL_INTERVAL)
    wait_timeout: float = field(default=DEFAULT_WAIT_TIMEOUT)
    max_concurrency: int = field(default=DEFAULT_MAX_CONCURRENCY)
    
    def get_api_key(self) -> str | None:
        """Get API key, falling back to environment variable."""
        return self.api_key or os.environ.get("AUDIXA_API_KEY")


# Global configuration instance
_config = GlobalConfig()


def get_config() -> GlobalConfig:
    """Get the global configuration instance."""
    return _config


def set_api_key(api_key: str) -> None:
    """
    Set the global API key for module-level functions.
    
    Args:
        api_key: Your Audixa API key.
        
    Example:
        >>> import audixa
        >>> audixa.set_api_key("your-api-key")
    """
    _config.api_key = api_key


def set_base_url(base_url: str) -> None:
    """
    Set the global API base URL.
    
    Args:
        base_url: The API base URL (e.g., "https://api.audixa.ai/v2").
        
    Example:
        >>> import audixa
        >>> audixa.set_base_url("https://api.audixa.ai/v2")
    """
    _config.base_url = base_url.rstrip("/")


def is_format_supported(format_name: str) -> bool:
    """
    Check if an audio format is currently supported.
    
    Args:
        format_name: The format to check (e.g., "wav", "mp3").
        
    Returns:
        True if the format is supported, False otherwise.
    """
    return format_name.lower() in SUPPORTED_FORMATS


def get_format_extension(format_name: str) -> str:
    """
    Get the file extension for a given audio format.
    
    Args:
        format_name: The audio format (e.g., "wav").
        
    Returns:
        The file extension including the dot (e.g., ".wav").
        
    Raises:
        ValueError: If the format is not supported.
    """
    ext = FORMAT_EXTENSIONS.get(format_name.lower())
    if ext is None:
        raise ValueError(
            f"Unsupported format: {format_name}. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )
    return ext
