"""
Audixa Synchronous Client.

Provides the AudixaClient class for synchronous API interactions.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import requests

from .config import (
    DEFAULT_BASE_URL,
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
    download_file_sync,
    logger,
    retry_sync,
    validate_output_filepath,
    validate_text,
)

# Type aliases for TTS parameters
Model = Literal["base", "advance"]
Emotion = Literal["neutral", "happy", "sad", "angry", "surprised"]


class AudixaClient:
    """
    Synchronous client for the Audixa TTS API.
    
    Use this client for synchronous (blocking) operations. For async operations,
    use AsyncAudixaClient instead.
    
    Args:
        api_key: Your Audixa API key. If not provided, falls back to
            the AUDIXA_API_KEY environment variable or global config.
        base_url: API base URL. Defaults to https://api.audixa.ai/v2.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries for failed requests.
    
    Example:
        >>> client = AudixaClient(api_key="your-api-key")
        >>> gen_id = client.tts("Hello, world!", voice="am_ethan")
        >>> status = client.status(gen_id)
        >>> if status["status"] == "completed":
        ...     print(status["audio_url"])
    
    Example (convenience method):
        >>> client = AudixaClient(api_key="your-api-key")
        >>> audio_url = client.tts_and_wait("Hello, world!", voice="am_ethan")
        >>> print(audio_url)
    
    Example (save to file):
        >>> client = AudixaClient(api_key="your-api-key")
        >>> filepath = client.tts_to_file("Hello!", "output.wav", voice="am_ethan")
        >>> print(f"Saved to {filepath}")
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """Initialize the Audixa client."""
        config = get_config()
        self._api_key = api_key or config.get_api_key()
        self._base_url = (base_url or config.base_url).rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._session: requests.Session | None = None
    
    @property
    def api_key(self) -> str | None:
        """Get the current API key."""
        return self._api_key
    
    @api_key.setter
    def api_key(self, value: str) -> None:
        """Set the API key."""
        self._api_key = value
    
    def _get_session(self) -> requests.Session:
        """Get or create the requests session."""
        if self._session is None:
            self._session = requests.Session()
        return self._session
    
    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authentication."""
        if not self._api_key:
            raise AuthenticationError(
                "No API key provided. Set it via AudixaClient(api_key=...), "
                "audixa.set_api_key(...), or the AUDIXA_API_KEY environment variable."
            )
        return {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
            "User-Agent": "audixa-python/0.1.0",
        }
    
    def _handle_response(
        self,
        response: requests.Response,
        accept_201: bool = False,
    ) -> dict[str, Any]:
        """
        Handle API response and raise appropriate exceptions.
        
        Args:
            response: The requests Response object.
            accept_201: If True, also accept 201 as a success status.
            
        Returns:
            Parsed JSON response data.
            
        Raises:
            AuthenticationError: For 401 responses.
            RateLimitError: For 429 responses.
            APIError: For other error responses.
        """
        try:
            data = response.json()
        except ValueError:
            data = {"error": response.text or "Unknown error"}
        
        success_codes = [200, 201] if accept_201 else [200]
        if response.status_code in success_codes:
            return data
        elif response.status_code == 401:
            raise AuthenticationError(
                message=data.get("error", "Authentication failed"),
                status_code=response.status_code,
                response_data=data,
            )
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                message=data.get("error", "Rate limit exceeded"),
                status_code=response.status_code,
                response_data=data,
                retry_after=float(retry_after) if retry_after else None,
            )
        else:
            raise APIError(
                message=data.get("error", f"API error: {response.status_code}"),
                status_code=response.status_code,
                response_data=data,
            )
    
    def _request(
        self,
        method: str,
        endpoint: str,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        accept_201: bool = False,
    ) -> dict[str, Any]:
        """
        Make an HTTP request with retry logic.
        
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
        session = self._get_session()
        
        logger.debug(f"Request: {method} {url}")
        
        @retry_sync(max_retries=self._max_retries)
        def do_request() -> dict[str, Any]:
            try:
                response = session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json,
                    params=params,
                    timeout=self._timeout,
                )
                return self._handle_response(response, accept_201=accept_201)
            except requests.exceptions.Timeout as e:
                raise TimeoutError(f"Request timed out after {self._timeout}s") from e
            except requests.exceptions.RequestException as e:
                raise NetworkError(f"Network error: {e}", original_error=e) from e
        
        return do_request()
    
    def tts(
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
        Generate speech from text.
        
        This method initiates TTS generation and returns immediately with a
        generation ID. Use status() to check progress, or tts_and_wait() for
        convenience.
        
        Args:
            text: The text to convert to speech. Must be at least 30 characters.
            voice: The Voice ID to use for generation (e.g., "am_ethan").
            model: The model to use: "base" or "advance". Defaults to "base".
            speed: Playback speed (0.5 to 2.0). Defaults to 1.0.
            emotion: (Advance model only) Emotion to apply: "neutral", "happy", 
                "sad", "angry", or "surprised". Defaults to "neutral".
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
            >>> client = AudixaClient(api_key="your-key")
            >>> gen_id = client.tts(
            ...     "Hello, world! Welcome to Audixa AI.",
            ...     voice="am_ethan",
            ...     model="base",
            ...     speed=1.1,
            ... )
            >>> print(gen_id)
        
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
        
        logger.info(f"Starting TTS generation for text: {text[:50]}...")
        response = self._request("POST", ENDPOINTS["tts"], json=payload, accept_201=True)
        
        generation_id = response.get("generation_id") or response.get("id")
        if not generation_id:
            raise APIError("No generation ID in response", response_data=response)
        
        logger.info(f"TTS generation started: {generation_id}")
        return generation_id
    
    def status(self, generation_id: str) -> dict[str, Any]:
        """
        Check the status of a TTS generation.
        
        Args:
            generation_id: The generation ID from tts().
            
        Returns:
            Status dictionary containing:
            - status: "Generating", "Completed", or "Failed"
            - audio_url: URL to download audio (when Completed)
            - error: Error message (when Failed)
            
        Example:
            >>> status = client.status("gen_abc123")
            >>> if status["status"] == "Completed":
            ...     print(status["audio_url"])
        """
        logger.debug(f"Checking status for generation: {generation_id}")
        return self._request(
            "GET",
            ENDPOINTS["status"],
            params={"generation_id": generation_id},
        )
    
    def tts_and_wait(
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
        Generate speech and wait for completion.
        
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
            >>> audio_url = client.tts_and_wait(
            ...     "Hello, welcome to Audixa AI!",
            ...     voice="am_ethan",
            ...     timeout=120,
            ... )
            >>> print(audio_url)
        """
        generation_id = self.tts(
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
        
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Generation timed out after {timeout}s",
                    timeout_seconds=timeout,
                )
            
            status_response = self.status(generation_id)
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
            time.sleep(poll_interval)
    
    def tts_to_file(
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
        Generate speech and save to a file.
        
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
            >>> filepath = client.tts_to_file(
            ...     "Hello, welcome to Audixa!",
            ...     "output.wav",
            ...     voice="am_ethan",
            ... )
            >>> print(f"Audio saved to: {filepath}")
        """
        # Validate output path
        output_path = validate_output_filepath(filepath)
        
        # Generate and wait for audio URL
        audio_url = self.tts_and_wait(
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
        download_file_sync(audio_url, output_path, timeout=self._timeout)
        
        return str(output_path)
    
    def list_voices(self, model: Model = "base") -> list[dict[str, Any]]:
        """
        List available voices for a specific model.
        
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
            >>> voices = client.list_voices(model="base")
            >>> for voice in voices:
            ...     print(f"{voice['voice_id']}: {voice['name']}")
        
        See Also:
            Audixa documentation: https://docs.audixa.ai/api/voices
        """
        logger.debug(f"Fetching available voices for model: {model}")
        response = self._request("GET", ENDPOINTS["voices"], params={"model": model})
        voices = response.get("voices", [])
        logger.info(f"Found {len(voices)} available voices for {model} model")
        return voices
    
    def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            self._session.close()
            self._session = None
    
    def __enter__(self) -> "AudixaClient":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
