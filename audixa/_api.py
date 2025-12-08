"""
Audixa SDK Module-Level API Functions.

Provides convenience functions that use global configuration and default clients.
These functions are exposed at the top level of the audixa package.
"""

from __future__ import annotations

from typing import Any, Literal

from .async_client import AsyncAudixaClient
from .client import AudixaClient
from .config import get_config

# Type aliases
Model = Literal["base", "advance"]
Emotion = Literal["neutral", "happy", "sad", "angry", "surprised"]

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
    voice: str,
    model: Model = "base",
    speed: float = 1.0,
    emotion: Emotion | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    do_sample: bool | None = None,
    **kwargs: Any,
) -> str:
    """
    Generate speech from text (synchronous).
    
    This initiates TTS generation and returns a generation ID.
    Use status() to check progress, or tts_and_wait() for convenience.
    
    Args:
        text: The text to convert to speech. Must be at least 30 characters.
        voice: The Voice ID (e.g., "am_ethan").
        model: "base" or "advance". Defaults to "base".
        speed: Playback speed (0.5 to 2.0). Defaults to 1.0.
        emotion: (Advance only) "neutral", "happy", "sad", "angry", "surprised".
        temperature: (Advance only) Randomness control (0.7-0.9).
        top_p: (Advance only) Nucleus sampling (0.7-0.98).
        do_sample: (Advance only) Enable sampling.
        
    Returns:
        The generation ID for tracking the request.
        
    Example:
        >>> import audixa
        >>> audixa.set_api_key("your-key")
        >>> gen_id = audixa.tts("Hello, world!", voice="am_ethan")
    """
    return _get_default_client().tts(
        text,
        voice=voice,
        model=model,
        speed=speed,
        emotion=emotion,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        **kwargs,
    )


def status(generation_id: str) -> dict[str, Any]:
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
    return _get_default_client().status(generation_id)


def tts_and_wait(
    text: str,
    voice: str,
    model: Model = "base",
    speed: float = 1.0,
    poll_interval: float | None = None,
    timeout: float | None = None,
    emotion: Emotion | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    do_sample: bool | None = None,
    **kwargs: Any,
) -> str:
    """
    Generate speech and wait for completion (synchronous).
    
    Args:
        text: The text to convert to speech.
        voice: The Voice ID (e.g., "am_ethan").
        model: "base" or "advance".
        speed: Playback speed (0.5 to 2.0).
        poll_interval: Time between status checks in seconds.
        timeout: Maximum time to wait in seconds.
        emotion: (Advance only) Emotion to apply.
        temperature: (Advance only) Randomness control.
        top_p: (Advance only) Nucleus sampling.
        do_sample: (Advance only) Enable sampling.
        
    Returns:
        The audio URL for the completed generation.
        
    Example:
        >>> audio_url = audixa.tts_and_wait("Hello!", voice="am_ethan")
    """
    config = get_config()
    return _get_default_client().tts_and_wait(
        text,
        voice=voice,
        model=model,
        speed=speed,
        poll_interval=poll_interval or config.poll_interval,
        timeout=timeout or config.wait_timeout,
        emotion=emotion,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        **kwargs,
    )


def tts_to_file(
    text: str,
    filepath: str,
    voice: str,
    model: Model = "base",
    speed: float = 1.0,
    poll_interval: float | None = None,
    timeout: float | None = None,
    emotion: Emotion | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    do_sample: bool | None = None,
    **kwargs: Any,
) -> str:
    """
    Generate speech and save to a file (synchronous).
    
    Args:
        text: The text to convert to speech.
        filepath: Output file path. Must have .wav extension.
        voice: The Voice ID (e.g., "am_ethan").
        model: "base" or "advance".
        speed: Playback speed (0.5 to 2.0).
        poll_interval: Time between status checks in seconds.
        timeout: Maximum time to wait in seconds.
        emotion: (Advance only) Emotion to apply.
        temperature: (Advance only) Randomness control.
        top_p: (Advance only) Nucleus sampling.
        do_sample: (Advance only) Enable sampling.
        
    Returns:
        The path to the saved audio file.
        
    Example:
        >>> audixa.tts_to_file("Hello!", "output.wav", voice="am_ethan")
    """
    config = get_config()
    return _get_default_client().tts_to_file(
        text,
        filepath,
        voice=voice,
        model=model,
        speed=speed,
        poll_interval=poll_interval or config.poll_interval,
        timeout=timeout or config.wait_timeout,
        emotion=emotion,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        **kwargs,
    )


def list_voices(model: Model = "base") -> list[dict[str, Any]]:
    """
    List available voices for a specific model (synchronous).
    
    Args:
        model: The model to fetch voices for: "base" or "advance".
    
    Returns:
        List of voice dictionaries with voice_id, name, gender, accent, etc.
        
    Example:
        >>> voices = audixa.list_voices(model="base")
        >>> for v in voices:
        ...     print(v["voice_id"], v["name"])
    """
    return _get_default_client().list_voices(model=model)


# =============================================================================
# Asynchronous API Functions
# =============================================================================

async def atts(
    text: str,
    voice: str,
    model: Model = "base",
    speed: float = 1.0,
    emotion: Emotion | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    do_sample: bool | None = None,
    **kwargs: Any,
) -> str:
    """
    Generate speech from text (asynchronous).
    
    Args:
        text: The text to convert to speech.
        voice: The Voice ID (e.g., "am_ethan").
        model: "base" or "advance".
        speed: Playback speed (0.5 to 2.0).
        emotion: (Advance only) Emotion to apply.
        temperature: (Advance only) Randomness control.
        top_p: (Advance only) Nucleus sampling.
        do_sample: (Advance only) Enable sampling.
        
    Returns:
        The generation ID for tracking the request.
        
    Example:
        >>> gen_id = await audixa.atts("Hello!", voice="am_ethan")
    """
    return await _get_default_async_client().tts(
        text,
        voice=voice,
        model=model,
        speed=speed,
        emotion=emotion,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        **kwargs,
    )


async def astatus(generation_id: str) -> dict[str, Any]:
    """
    Check the status of a TTS generation (asynchronous).
    
    Args:
        generation_id: The generation ID from atts().
        
    Returns:
        Status dictionary with 'status', 'audio_url', etc.
        
    Example:
        >>> status = await audixa.astatus("gen_abc123")
    """
    return await _get_default_async_client().status(generation_id)


async def atts_and_wait(
    text: str,
    voice: str,
    model: Model = "base",
    speed: float = 1.0,
    poll_interval: float | None = None,
    timeout: float | None = None,
    emotion: Emotion | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    do_sample: bool | None = None,
    **kwargs: Any,
) -> str:
    """
    Generate speech and wait for completion (asynchronous).
    
    Args:
        text: The text to convert to speech.
        voice: The Voice ID (e.g., "am_ethan").
        model: "base" or "advance".
        speed: Playback speed (0.5 to 2.0).
        poll_interval: Time between status checks in seconds.
        timeout: Maximum time to wait in seconds.
        emotion: (Advance only) Emotion to apply.
        temperature: (Advance only) Randomness control.
        top_p: (Advance only) Nucleus sampling.
        do_sample: (Advance only) Enable sampling.
        
    Returns:
        The audio URL for the completed generation.
        
    Example:
        >>> audio_url = await audixa.atts_and_wait("Hello!", voice="am_ethan")
    """
    config = get_config()
    return await _get_default_async_client().tts_and_wait(
        text,
        voice=voice,
        model=model,
        speed=speed,
        poll_interval=poll_interval or config.poll_interval,
        timeout=timeout or config.wait_timeout,
        emotion=emotion,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        **kwargs,
    )


async def atts_to_file(
    text: str,
    filepath: str,
    voice: str,
    model: Model = "base",
    speed: float = 1.0,
    poll_interval: float | None = None,
    timeout: float | None = None,
    emotion: Emotion | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    do_sample: bool | None = None,
    **kwargs: Any,
) -> str:
    """
    Generate speech and save to a file (asynchronous).
    
    Args:
        text: The text to convert to speech.
        filepath: Output file path. Must have .wav extension.
        voice: The Voice ID (e.g., "am_ethan").
        model: "base" or "advance".
        speed: Playback speed (0.5 to 2.0).
        poll_interval: Time between status checks in seconds.
        timeout: Maximum time to wait in seconds.
        emotion: (Advance only) Emotion to apply.
        temperature: (Advance only) Randomness control.
        top_p: (Advance only) Nucleus sampling.
        do_sample: (Advance only) Enable sampling.
        
    Returns:
        The path to the saved audio file.
        
    Example:
        >>> await audixa.atts_to_file("Hello!", "output.wav", voice="am_ethan")
    """
    config = get_config()
    return await _get_default_async_client().tts_to_file(
        text,
        filepath,
        voice=voice,
        model=model,
        speed=speed,
        poll_interval=poll_interval or config.poll_interval,
        timeout=timeout or config.wait_timeout,
        emotion=emotion,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        **kwargs,
    )


async def alist_voices(model: Model = "base") -> list[dict[str, Any]]:
    """
    List available voices for a specific model (asynchronous).
    
    Args:
        model: The model to fetch voices for: "base" or "advance".
    
    Returns:
        List of voice dictionaries with voice_id, name, gender, accent, etc.
        
    Example:
        >>> voices = await audixa.alist_voices(model="advance")
    """
    return await _get_default_async_client().list_voices(model=model)
