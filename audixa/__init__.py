"""
Audixa - Official Python SDK for Audixa AI Text-to-Speech API.

A production-ready SDK providing both synchronous and asynchronous interfaces
for the Audixa TTS API.

Quick Start:
    >>> import audixa
    >>> audixa.set_api_key("your-api-key")
    >>> audio_url = audixa.tts_and_wait("Hello, world!", voice_id="emma")
    >>> print(audio_url)

Async Usage:
    >>> import audixa
    >>> audixa.set_api_key("your-api-key")
    >>> audio_url = await audixa.atts_and_wait("Hello, world!", voice_id="emma")

Advanced Usage (Client Classes):
    >>> from audixa import AudixaClient
    >>> client = AudixaClient(api_key="your-key")
    >>> with client:
    ...     audio_url = client.tts_and_wait("Hello!", voice_id="emma")

For more information, see: https://docs.audixa.ai
"""

from __future__ import annotations

from .config import SDK_VERSION as __version__
__author__ = "Audixa AI"

# =============================================================================
# Public API Exports
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Configuration functions
    "set_api_key",
    "set_base_url",
    "set_custom_endpoint_slug",
    # Synchronous API
    "tts",
    "status",
    "tts_and_wait",
    "tts_to_file",
    "list_voices",
    "history",
    # Asynchronous API
    "atts",
    "astatus",
    "atts_and_wait",
    "atts_to_file",
    "alist_voices",
    "ahistory",
    # Client classes
    "AudixaClient",
    "AsyncAudixaClient",
    # Exceptions
    "AudixaError",
    "AuthenticationError",
    "RateLimitError",
    "APIError",
    "NetworkError",
    "TimeoutError",
    "GenerationError",
    "UnsupportedFormatError",
    "ValidationError",
]

# =============================================================================
# Imports
# =============================================================================

# Configuration
from .config import set_api_key, set_base_url, set_custom_endpoint_slug

# Client classes
from .client import AudixaClient
from .async_client import AsyncAudixaClient

# Exceptions
from .exceptions import (
    AudixaError,
    AuthenticationError,
    RateLimitError,
    APIError,
    NetworkError,
    TimeoutError,
    GenerationError,
    UnsupportedFormatError,
    ValidationError,
)

# Module-level API functions
from ._api import (
    # Sync
    tts,
    status,
    tts_and_wait,
    tts_to_file,
    list_voices,
    history,
    # Async
    atts,
    astatus,
    atts_and_wait,
    atts_to_file,
    alist_voices,
    ahistory,
)
