"""
Audixa SDK Module-Level API Functions.

Provides convenience functions that use global configuration and default clients.
These functions are exposed at the top level of the audixa package.
"""

from __future__ import annotations

from typing import Any, Literal

from .async_client import AsyncAudixaClient
from .client import AudixaClient
from .config import AudioFormat, get_config

# Type aliases
Model = Literal["base", "advanced"]

# =============================================================================
# Default Client Singletons
# =============================================================================

_default_client: AudixaClient | None = None
_default_async_client: AsyncAudixaClient | None = None


def _get_default_client() -> AudixaClient:
    """Get or create the default synchronous client."""
    global _default_client
    if _default_client is None:
        _default_client = AudixaClient()
    return _default_client


def _get_default_async_client() -> AsyncAudixaClient:
    """Get or create the default asynchronous client."""
    global _default_async_client
    if _default_async_client is None:
        _default_async_client = AsyncAudixaClient()
    return _default_async_client


# =============================================================================
# Synchronous API Functions
# =============================================================================

def tts(
    text: str,
    voice_id: str | None = None,
    model: Model = "base",
    speed: float = 1.0,
    cfg_weight: float | None = None,
    exaggeration: float | None = None,
    audio_format: AudioFormat | None = None,
    custom_endpoint_slug: str | None = None,
    **kwargs: Any,
) -> str:
    """
    Generate speech from text (synchronous).
    
    This initiates TTS generation and returns a generation ID.
    Use status() to check progress, or tts_and_wait() for convenience.
    
    Args:
        text: The text to convert to speech.
        voice_id: The Voice ID (e.g., "emma").
        model: "base" or "advanced". Defaults to "base".
        speed: Playback speed (0.5 to 2.0). Defaults to 1.0.
        cfg_weight: (Advanced only) CFG weight.
        exaggeration: (Advanced only) Exaggeration.
        audio_format: Output format ("wav" or "mp3").
        custom_endpoint_slug: Optional custom endpoint slug to route this request.
        
    Returns:
        The generation ID for tracking the request.
        
    Example:
        >>> import audixa
        >>> audixa.set_api_key("your-key")
        >>> gen_id = audixa.tts("Hello, world!", voice_id="emma")
    """
    return _get_default_client().tts(
        text,
        voice_id=voice_id,
        model=model,
        speed=speed,
        cfg_weight=cfg_weight,
        exaggeration=exaggeration,
        audio_format=audio_format,
        custom_endpoint_slug=custom_endpoint_slug,
        **kwargs,
    )


def status(generation_id: str, custom_endpoint_slug: str | None = None) -> dict[str, Any]:
    """
    Check the status of a TTS generation (synchronous).
    
    Args:
        generation_id: The generation ID from tts().
        
    Returns:
        Status dictionary with 'status', 'audio_url', etc.
        
    Example:
        >>> status = audixa.status("gen_abc123")
        >>> print(status["status"])
    """
    return _get_default_client().status(generation_id, custom_endpoint_slug=custom_endpoint_slug)


def tts_and_wait(
    text: str,
    voice_id: str | None = None,
    model: Model = "base",
    speed: float = 1.0,
    poll_interval: float | None = None,
    timeout: float | None = None,
    cfg_weight: float | None = None,
    exaggeration: float | None = None,
    audio_format: AudioFormat | None = None,
    custom_endpoint_slug: str | None = None,
    **kwargs: Any,
) -> str:
    """
    Generate speech and wait for completion (synchronous).
    
    Args:
        text: The text to convert to speech.
        voice_id: The Voice ID (e.g., "emma").
        model: "base" or "advanced".
        speed: Playback speed (0.5 to 2.0).
        poll_interval: Time between status checks in seconds.
        timeout: Maximum time to wait in seconds.
        cfg_weight: (Advanced only) CFG weight.
        exaggeration: (Advanced only) Exaggeration.
        audio_format: Output format ("wav" or "mp3").
        custom_endpoint_slug: Optional custom endpoint slug to route this request.
        
    Returns:
        The audio URL for the completed generation.
        
    Example:
        >>> audio_url = audixa.tts_and_wait("Hello!", voice_id="emma")
    """
    config = get_config()
    return _get_default_client().tts_and_wait(
        text,
        voice_id=voice_id,
        model=model,
        speed=speed,
        poll_interval=poll_interval or config.poll_interval,
        timeout=timeout or config.wait_timeout,
        cfg_weight=cfg_weight,
        exaggeration=exaggeration,
        audio_format=audio_format,
        custom_endpoint_slug=custom_endpoint_slug,
        **kwargs,
    )


def tts_to_file(
    text: str,
    filepath: str,
    voice_id: str | None = None,
    model: Model = "base",
    speed: float = 1.0,
    poll_interval: float | None = None,
    timeout: float | None = None,
    cfg_weight: float | None = None,
    exaggeration: float | None = None,
    audio_format: AudioFormat | None = None,
    custom_endpoint_slug: str | None = None,
    **kwargs: Any,
) -> str:
    """
    Generate speech and save to a file (synchronous).
    
    Args:
        text: The text to convert to speech.
        filepath: Output file path.
        voice_id: The Voice ID (e.g., "emma").
        model: "base" or "advanced".
        speed: Playback speed (0.5 to 2.0).
        poll_interval: Time between status checks in seconds.
        timeout: Maximum time to wait in seconds.
        cfg_weight: (Advanced only) CFG weight.
        exaggeration: (Advanced only) Exaggeration.
        audio_format: Output format ("wav" or "mp3").
        custom_endpoint_slug: Optional custom endpoint slug to route this request.
        
    Returns:
        The path to the saved audio file.
        
    Example:
        >>> audixa.tts_to_file("Hello!", "output.wav", voice_id="emma")
    """
    config = get_config()
    return _get_default_client().tts_to_file(
        text,
        filepath,
        voice_id=voice_id,
        model=model,
        speed=speed,
        poll_interval=poll_interval or config.poll_interval,
        timeout=timeout or config.wait_timeout,
        cfg_weight=cfg_weight,
        exaggeration=exaggeration,
        audio_format=audio_format,
        custom_endpoint_slug=custom_endpoint_slug,
        **kwargs,
    )


def list_voices(
    model: Model | None = None,
    limit: int = 100,
    offset: int = 0,
    include_metadata: bool = False,
) -> list[dict[str, Any]] | dict[str, Any]:
    """
    List available voices for a specific model (synchronous).
    
    Args:
        model: Optional model filter: "base" or "advanced".
        limit: Maximum number of results to return (1-500).
        offset: Number of results to skip.
        include_metadata: If True, return the full response with pagination.
    
    Returns:
        List of voice dictionaries with voice_id, name, gender, accent, etc.
        
    Example:
        >>> voices = audixa.list_voices(model="base")
        >>> for v in voices:
        ...     print(v["voice_id"], v["name"])
    """
    return _get_default_client().list_voices(
        model=model,
        limit=limit,
        offset=offset,
        include_metadata=include_metadata,
    )


def history(
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
    include_metadata: bool = False,
) -> list[dict[str, Any]] | dict[str, Any]:
    """
    Retrieve generation history (synchronous).
    
    Args:
        limit: Maximum number of results to return (1-100).
        offset: Number of results to skip.
        status: Optional status filter (IN_QUEUE, GENERATING, COMPLETED, FAILED, EXPIRED).
        include_metadata: If True, return the full response with pagination.
    """
    return _get_default_client().history(
        limit=limit,
        offset=offset,
        status=status,
        include_metadata=include_metadata,
    )


# =============================================================================
# Asynchronous API Functions
# =============================================================================

async def atts(
    text: str,
    voice_id: str | None = None,
    model: Model = "base",
    speed: float = 1.0,
    cfg_weight: float | None = None,
    exaggeration: float | None = None,
    audio_format: AudioFormat | None = None,
    custom_endpoint_slug: str | None = None,
    **kwargs: Any,
) -> str:
    """
    Generate speech from text (asynchronous).
    
    Args:
        text: The text to convert to speech.
        voice_id: The Voice ID (e.g., "emma").
        model: "base" or "advanced".
        speed: Playback speed (0.5 to 2.0).
        cfg_weight: (Advanced only) CFG weight.
        exaggeration: (Advanced only) Exaggeration.
        audio_format: Output format ("wav" or "mp3").
        custom_endpoint_slug: Optional custom endpoint slug to route this request.
        
    Returns:
        The generation ID for tracking the request.
        
    Example:
        >>> gen_id = await audixa.atts("Hello!", voice_id="emma")
    """
    return await _get_default_async_client().tts(
        text,
        voice_id=voice_id,
        model=model,
        speed=speed,
        cfg_weight=cfg_weight,
        exaggeration=exaggeration,
        audio_format=audio_format,
        custom_endpoint_slug=custom_endpoint_slug,
        **kwargs,
    )


async def astatus(
    generation_id: str,
    custom_endpoint_slug: str | None = None,
) -> dict[str, Any]:
    """
    Check the status of a TTS generation (asynchronous).
    
    Args:
        generation_id: The generation ID from atts().
        
    Returns:
        Status dictionary with 'status', 'audio_url', etc.
        
    Example:
        >>> status = await audixa.astatus("gen_abc123")
    """
    return await _get_default_async_client().status(
        generation_id, custom_endpoint_slug=custom_endpoint_slug
    )


async def atts_and_wait(
    text: str,
    voice_id: str | None = None,
    model: Model = "base",
    speed: float = 1.0,
    poll_interval: float | None = None,
    timeout: float | None = None,
    cfg_weight: float | None = None,
    exaggeration: float | None = None,
    audio_format: AudioFormat | None = None,
    custom_endpoint_slug: str | None = None,
    **kwargs: Any,
) -> str:
    """
    Generate speech and wait for completion (asynchronous).
    
    Args:
        text: The text to convert to speech.
        voice_id: The Voice ID (e.g., "emma").
        model: "base" or "advanced".
        speed: Playback speed (0.5 to 2.0).
        poll_interval: Time between status checks in seconds.
        timeout: Maximum time to wait in seconds.
        cfg_weight: (Advanced only) CFG weight.
        exaggeration: (Advanced only) Exaggeration.
        audio_format: Output format ("wav" or "mp3").
        custom_endpoint_slug: Optional custom endpoint slug to route this request.
        
    Returns:
        The audio URL for the completed generation.
        
    Example:
        >>> audio_url = await audixa.atts_and_wait("Hello!", voice_id="emma")
    """
    config = get_config()
    return await _get_default_async_client().tts_and_wait(
        text,
        voice_id=voice_id,
        model=model,
        speed=speed,
        poll_interval=poll_interval or config.poll_interval,
        timeout=timeout or config.wait_timeout,
        cfg_weight=cfg_weight,
        exaggeration=exaggeration,
        audio_format=audio_format,
        custom_endpoint_slug=custom_endpoint_slug,
        **kwargs,
    )


async def atts_to_file(
    text: str,
    filepath: str,
    voice_id: str | None = None,
    model: Model = "base",
    speed: float = 1.0,
    poll_interval: float | None = None,
    timeout: float | None = None,
    cfg_weight: float | None = None,
    exaggeration: float | None = None,
    audio_format: AudioFormat | None = None,
    custom_endpoint_slug: str | None = None,
    **kwargs: Any,
) -> str:
    """
    Generate speech and save to a file (asynchronous).
    
    Args:
        text: The text to convert to speech.
        filepath: Output file path.
        voice_id: The Voice ID (e.g., "emma").
        model: "base" or "advanced".
        speed: Playback speed (0.5 to 2.0).
        poll_interval: Time between status checks in seconds.
        timeout: Maximum time to wait in seconds.
        cfg_weight: (Advanced only) CFG weight.
        exaggeration: (Advanced only) Exaggeration.
        audio_format: Output format ("wav" or "mp3").
        custom_endpoint_slug: Optional custom endpoint slug to route this request.
        
    Returns:
        The path to the saved audio file.
        
    Example:
        >>> await audixa.atts_to_file("Hello!", "output.wav", voice_id="emma")
    """
    config = get_config()
    return await _get_default_async_client().tts_to_file(
        text,
        filepath,
        voice_id=voice_id,
        model=model,
        speed=speed,
        poll_interval=poll_interval or config.poll_interval,
        timeout=timeout or config.wait_timeout,
        cfg_weight=cfg_weight,
        exaggeration=exaggeration,
        audio_format=audio_format,
        custom_endpoint_slug=custom_endpoint_slug,
        **kwargs,
    )


async def alist_voices(
    model: Model | None = None,
    limit: int = 100,
    offset: int = 0,
    include_metadata: bool = False,
) -> list[dict[str, Any]] | dict[str, Any]:
    """
    List available voices for a specific model (asynchronous).
    
    Args:
        model: Optional model filter: "base" or "advanced".
        limit: Maximum number of results to return (1-500).
        offset: Number of results to skip.
        include_metadata: If True, return the full response with pagination.
    
    Returns:
        List of voice dictionaries with voice_id, name, gender, accent, etc.
        
    Example:
        >>> voices = await audixa.alist_voices(model="advanced")
    """
    return await _get_default_async_client().list_voices(
        model=model,
        limit=limit,
        offset=offset,
        include_metadata=include_metadata,
    )


async def ahistory(
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
    include_metadata: bool = False,
) -> list[dict[str, Any]] | dict[str, Any]:
    """
    Retrieve generation history (asynchronous).
    
    Args:
        limit: Maximum number of results to return (1-100).
        offset: Number of results to skip.
        status: Optional status filter (IN_QUEUE, GENERATING, COMPLETED, FAILED, EXPIRED).
        include_metadata: If True, return the full response with pagination.
    """
    return await _get_default_async_client().history(
        limit=limit,
        offset=offset,
        status=status,
        include_metadata=include_metadata,
    )

