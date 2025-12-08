"""
Audixa Asynchronous Client.

Provides the AsyncAudixaClient class for async API interactions.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import aiohttp

from .config import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_TIMEOUT,
    DEFAULT_WAIT_TIMEOUT,
    ENDPOINTS,
    AudioFormat,
    get_config,
)
from .exceptions import (
    APIError,
    AuthenticationError,
    GenerationError,
    NetworkError,
    RateLimitError,
    TimeoutError,
)
from .utils import (
    build_url,
    download_file_async,
    logger,
    retry_async_operation,
    validate_output_filepath,
    validate_text,
)

# Type aliases for TTS parameters
Model = Literal["base", "advance"]
Emotion = Literal["neutral", "happy", "sad", "angry", "surprised"]


class AsyncAudixaClient:
    """
    Asynchronous client for the Audixa TTS API.
    
    Use this client for async (non-blocking) operations with asyncio.
    For synchronous operations, use AudixaClient instead.
    
    Args:
        api_key: Your Audixa API key. If not provided, falls back to
            the AUDIXA_API_KEY environment variable or global config.
        base_url: API base URL. Defaults to https://api.audixa.ai/v2.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries for failed requests.
        max_concurrency: Maximum concurrent requests (semaphore limit).
    
    Example:
        >>> async with AsyncAudixaClient(api_key="your-key") as client:
        ...     gen_id = await client.tts("Hello, world!", voice="am_ethan")
        ...     status = await client.status(gen_id)
        ...     print(status)
    
    Example (convenience method):
        >>> async with AsyncAudixaClient(api_key="your-key") as client:
        ...     audio_url = await client.tts_and_wait("Hello!", voice="am_ethan")
        ...     print(audio_url)
    
    Example (concurrent generation):
        >>> async with AsyncAudixaClient(api_key="your-key") as client:
        ...     tasks = [client.tts_and_wait(text, voice="am_ethan") for text in texts]
        ...     urls = await asyncio.gather(*tasks)
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        """Initialize the async Audixa client."""
        config = get_config()
        self._api_key = api_key or config.get_api_key()
        self._base_url = (base_url or config.base_url).rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._session: aiohttp.ClientSession | None = None
        self._owns_session = False
    
    @property
    def api_key(self) -> str | None:
        """Get the current API key."""
        return self._api_key
    
    @api_key.setter
    def api_key(self, value: str) -> None:
        """Set the API key."""
        self._api_key = value
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._owns_session = True
        return self._session
    
    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authentication."""
        if not self._api_key:
            raise AuthenticationError(
                "No API key provided. Set it via AsyncAudixaClient(api_key=...), "
                "audixa.set_api_key(...), or the AUDIXA_API_KEY environment variable."
            )
        return {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
            "User-Agent": "audixa-python/0.1.0",
        }
    
    async def _handle_response(
        self,
        response: aiohttp.ClientResponse,
        accept_201: bool = False,
    ) -> dict[str, Any]:
        """
        Handle API response and raise appropriate exceptions.
        
        Args:
            response: The aiohttp ClientResponse object.
            accept_201: If True, also accept 201 as a success status.
            
        Returns:
            Parsed JSON response data.
            
        Raises:
            AuthenticationError: For 401 responses.
            RateLimitError: For 429 responses.
            APIError: For other error responses.
        """
        try:
            data = await response.json()
        except (ValueError, aiohttp.ContentTypeError):
            text = await response.text()
            data = {"error": text or "Unknown error"}
        
        success_codes = [200, 201] if accept_201 else [200]
        if response.status in success_codes:
            return data
        elif response.status == 401:
            raise AuthenticationError(
                message=data.get("error", "Authentication failed"),
                status_code=response.status,
                response_data=data,
            )
        elif response.status == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                message=data.get("error", "Rate limit exceeded"),
                status_code=response.status,
                response_data=data,
                retry_after=float(retry_after) if retry_after else None,
            )
        else:
            raise APIError(
                message=data.get("error", f"API error: {response.status}"),
                status_code=response.status,
                response_data=data,
            )
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        accept_201: bool = False,
    ) -> dict[str, Any]:
        """
        Make an HTTP request with retry logic and rate limiting.
        
        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: API endpoint path.
            json: JSON body for POST requests.
            params: Query parameters for GET requests.
            accept_201: If True, accept 201 as a success status.
            
        Returns:
            Parsed JSON response.
        """
        url = f"{self._base_url}{endpoint}"
        headers = self._get_headers()
        session = await self._get_session()
        
        logger.debug(f"Async request: {method} {url}")
        
        async def do_request() -> dict[str, Any]:
            async with self._semaphore:
                try:
                    async with session.request(
                        method=method,
                        url=url,
                        headers=headers,
                        json=json,
                        params=params,
                    ) as response:
                        return await self._handle_response(response, accept_201=accept_201)
                except asyncio.TimeoutError as e:
                    raise TimeoutError(
                        f"Request timed out after {self._timeout}s"
                    ) from e
                except aiohttp.ClientError as e:
                    raise NetworkError(f"Network error: {e}", original_error=e) from e
        
        return await retry_async_operation(
            do_request,
            max_retries=self._max_retries,
        )
    
    async def tts(
        self,
        text: str,
        voice: str,
        model: Model = "base",
        speed: float = 1.0,
        # Advance model parameters
        emotion: Emotion | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate speech from text asynchronously.
        
        This method initiates TTS generation and returns immediately with a
        generation ID. Use status() to check progress, or tts_and_wait() for
        convenience.
        
        Args:
            text: The text to convert to speech. Must be at least 30 characters.
            voice: The Voice ID to use for generation (e.g., "am_ethan").
            model: The model to use: "base" or "advance". Defaults to "base".
            speed: Playback speed (0.5 to 2.0). Defaults to 1.0.
            emotion: (Advance model only) Emotion to apply: "neutral", "happy",
                "sad", "angry", or "surprised".
            temperature: (Advance model only) Controls randomness (0.7 to 0.9).
            top_p: (Advance model only) Nucleus sampling (0.7 to 0.98).
            do_sample: (Advance model only) Enable sampling for varied outputs.
            **kwargs: Additional parameters passed to the API.
            
        Returns:
            The generation ID for tracking the request.
            
        Raises:
            ValidationError: If the text is invalid.
            AuthenticationError: If the API key is invalid.
            RateLimitError: If rate limit is exceeded.
            APIError: For other API errors.
            NetworkError: For network-related errors.
            
        Example:
            >>> async with AsyncAudixaClient(api_key="your-key") as client:
            ...     gen_id = await client.tts(
            ...         "Hello, welcome to Audixa AI!",
            ...         voice="am_ethan",
            ...     )
            ...     print(gen_id)
        
        See Also:
            Voice Library: https://docs.audixa.ai/voices
            Models Guide: https://docs.audixa.ai/models
        """
        text = validate_text(text)
        
        payload: dict[str, Any] = {
            "text": text,
            "voice": voice,
            "model": model,
            "speed": speed,
        }
        
        # Add advance model parameters if provided
        if model == "advance":
            if emotion is not None:
                payload["emotion"] = emotion
            if temperature is not None:
                payload["temperature"] = temperature
            if top_p is not None:
                payload["top_p"] = top_p
            if do_sample is not None:
                payload["do_sample"] = do_sample
        
        payload.update(kwargs)
        
        logger.info(f"Starting async TTS generation for text: {text[:50]}...")
        response = await self._request("POST", ENDPOINTS["tts"], json=payload, accept_201=True)
        
        generation_id = response.get("generation_id") or response.get("id")
        if not generation_id:
            raise APIError("No generation ID in response", response_data=response)
        
        logger.info(f"TTS generation started: {generation_id}")
        return generation_id
    
    async def status(self, generation_id: str) -> dict[str, Any]:
        """
        Check the status of a TTS generation asynchronously.
        
        Args:
            generation_id: The generation ID from tts().
            
        Returns:
            Status dictionary containing:
            - status: "Generating", "Completed", or "Failed"
            - audio_url: URL to download audio (when Completed)
            - error: Error message (when Failed)
            
        Example:
            >>> async with AsyncAudixaClient(api_key="your-key") as client:
            ...     status = await client.status("gen_abc123")
            ...     print(status)
        """
        logger.debug(f"Checking status for generation: {generation_id}")
        return await self._request(
            "GET",
            ENDPOINTS["status"],
            params={"generation_id": generation_id},
        )
    
    async def tts_and_wait(
        self,
        text: str,
        voice: str,
        model: Model = "base",
        speed: float = 1.0,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        # Advance model parameters
        emotion: Emotion | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate speech and wait for completion asynchronously.
        
        This is a convenience method that combines tts() and status() polling.
        
        Args:
            text: The text to convert to speech. Must be at least 30 characters.
            voice: The Voice ID to use for generation.
            model: The model to use: "base" or "advance".
            speed: Playback speed (0.5 to 2.0).
            poll_interval: Time between status checks in seconds.
            timeout: Maximum time to wait for completion in seconds.
            emotion: (Advance model only) Emotion to apply.
            temperature: (Advance model only) Controls randomness.
            top_p: (Advance model only) Nucleus sampling.
            do_sample: (Advance model only) Enable sampling.
            **kwargs: Additional parameters passed to the API.
            
        Returns:
            The audio URL for the completed generation.
            
        Raises:
            TimeoutError: If generation doesn't complete within timeout.
            GenerationError: If generation fails.
            
        Example:
            >>> async with AsyncAudixaClient(api_key="your-key") as client:
            ...     audio_url = await client.tts_and_wait(
            ...         "Hello, welcome to Audixa!",
            ...         voice="am_ethan",
            ...     )
            ...     print(audio_url)
        """
        generation_id = await self.tts(
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
        
        start_time = asyncio.get_event_loop().time()
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Generation timed out after {timeout}s",
                    timeout_seconds=timeout,
                )
            
            status_response = await self.status(generation_id)
            status = status_response.get("status", "unknown")
            
            logger.debug(f"Generation {generation_id} status: {status}")
            
            if status == "Completed":
                audio_url = status_response.get("url") or status_response.get("audio_url")
                if not audio_url:
                    raise GenerationError(
                        "Generation completed but no audio URL provided",
                        generation_id=generation_id,
                        response_data=status_response,
                    )
                logger.info(f"Generation completed: {audio_url}")
                return audio_url
            
            elif status == "Failed":
                error_msg = status_response.get("error", "Unknown error")
                raise GenerationError(
                    f"Generation failed: {error_msg}",
                    generation_id=generation_id,
                    response_data=status_response,
                )
            
            # Still pending/processing, wait and retry
            await asyncio.sleep(poll_interval)
    
    async def tts_to_file(
        self,
        text: str,
        filepath: str,
        voice: str,
        model: Model = "base",
        speed: float = 1.0,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        # Advance model parameters
        emotion: Emotion | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate speech and save to a file asynchronously.
        
        This combines tts_and_wait() with audio download.
        
        Args:
            text: The text to convert to speech.
            filepath: Output file path. Must have .wav extension.
            voice: The Voice ID to use for generation.
            model: The model to use: "base" or "advance".
            speed: Playback speed (0.5 to 2.0).
            poll_interval: Time between status checks in seconds.
            timeout: Maximum time to wait for completion in seconds.
            emotion: (Advance model only) Emotion to apply.
            temperature: (Advance model only) Controls randomness.
            top_p: (Advance model only) Nucleus sampling.
            do_sample: (Advance model only) Enable sampling.
            **kwargs: Additional parameters passed to the API.
            
        Returns:
            The path to the saved audio file.
            
        Raises:
            UnsupportedFormatError: If file extension is not .wav.
            TimeoutError: If generation or download times out.
            GenerationError: If generation fails.
            NetworkError: If download fails.
            
        Example:
            >>> async with AsyncAudixaClient(api_key="your-key") as client:
            ...     filepath = await client.tts_to_file(
            ...         "Hello, welcome to Audixa!",
            ...         "output.wav",
            ...         voice="am_ethan",
            ...     )
            ...     print(f"Saved to {filepath}")
        """
        # Validate output path
        output_path = validate_output_filepath(filepath)
        
        # Generate and wait for audio URL
        audio_url = await self.tts_and_wait(
            text,
            voice=voice,
            model=model,
            speed=speed,
            poll_interval=poll_interval,
            timeout=timeout,
            emotion=emotion,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            **kwargs,
        )
        
        # Download to file
        await download_file_async(audio_url, output_path, timeout=self._timeout)
        
        return str(output_path)
    
    async def list_voices(self, model: Model = "base") -> list[dict[str, Any]]:
        """
        List available voices for a specific model asynchronously.
        
        Args:
            model: The model to fetch voices for: "base" or "advance".
        
        Returns:
            List of voice dictionaries, each containing:
            - voice_id: Voice ID for use in tts()
            - name: Display name
            - gender: Voice gender (e.g., "Male", "Female")
            - accent: Voice accent (e.g., "American", "British")
            - free: Whether the voice is free tier
            - description: Voice description
            
        Example:
            >>> async with AsyncAudixaClient(api_key="your-key") as client:
            ...     voices = await client.list_voices(model="base")
            ...     for voice in voices:
            ...         print(f"{voice['voice_id']}: {voice['name']}")
        
        See Also:
            Audixa documentation: https://docs.audixa.ai/api/voices
        """
        logger.debug(f"Fetching available voices for model: {model}")
        response = await self._request("GET", ENDPOINTS["voices"], params={"model": model})
        voices = response.get("voices", [])
        logger.info(f"Found {len(voices)} available voices for {model} model")
        return voices
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and self._owns_session:
            await self._session.close()
            self._session = None
    
    async def __aenter__(self) -> "AsyncAudixaClient":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
