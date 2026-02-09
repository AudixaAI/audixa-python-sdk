"""
Audixa Synchronous Client.

Provides the AudixaClient class for synchronous API interactions (v3).
"""

from __future__ import annotations

import time
from typing import Any, Literal

import requests

from .config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_TIMEOUT,
    DEFAULT_WAIT_TIMEOUT,
    ENDPOINTS,
    SDK_VERSION,
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
    ValidationError,
)
from .utils import (
    download_file_sync,
    logger,
    retry_sync,
    validate_output_filepath,
    validate_text,
)

# Type aliases for TTS parameters
Model = Literal["base", "advanced"]


class AudixaClient:
    """
    Synchronous client for the Audixa TTS API.
    
    Use this client for synchronous (blocking) operations. For async operations,
    use AsyncAudixaClient instead.
    
    Args:
        api_key: Your Audixa API key. If not provided, falls back to
            the AUDIXA_API_KEY environment variable or global config.
        base_url: API base URL. Defaults to https://api.audixa.ai.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries for failed requests.
        custom_endpoint_slug: Optional custom endpoint slug for dedicated routing.
    
    Example:
        >>> client = AudixaClient(api_key="your-api-key")
        >>> gen_id = client.tts("Hello, world!", voice_id="emma")
        >>> status = client.status(gen_id)
        >>> if status["status"] == "completed":
        ...     print(status["audio_url"])
    
    Example (convenience method):
        >>> client = AudixaClient(api_key="your-api-key")
        >>> audio_url = client.tts_and_wait("Hello, world!", voice_id="emma")
        >>> print(audio_url)
    
    Example (save to file):
        >>> client = AudixaClient(api_key="your-api-key")
        >>> filepath = client.tts_to_file("Hello!", "output.wav", voice_id="emma")
        >>> print(f"Saved to {filepath}")
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        custom_endpoint_slug: str | None = None,
    ) -> None:
        """Initialize the Audixa client."""
        config = get_config()
        self._api_key = api_key or config.get_api_key()
        self._base_url = (base_url or config.base_url).rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._custom_endpoint_slug = custom_endpoint_slug or config.custom_endpoint_slug
        self._session: requests.Session | None = None
    
    @property
    def api_key(self) -> str | None:
        """Get the current API key."""
        return self._api_key
    
    @api_key.setter
    def api_key(self, value: str) -> None:
        """Set the API key."""
        self._api_key = value

    @property
    def custom_endpoint_slug(self) -> str | None:
        """Get the custom endpoint slug (if enabled)."""
        return self._custom_endpoint_slug

    @custom_endpoint_slug.setter
    def custom_endpoint_slug(self, value: str | None) -> None:
        """Set or clear the custom endpoint slug."""
        self._custom_endpoint_slug = value or None
    
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
            "X-API-Key": self._api_key,
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"audixa-python/{SDK_VERSION}",
        }

    def _normalize_model(self, model: str) -> str:
        """Normalize model names for API compatibility."""
        if model == "advance":
            return "advanced"
        return model

    def _resolve_custom_slug(self, custom_endpoint_slug: str | None) -> str | None:
        if custom_endpoint_slug is not None:
            return custom_endpoint_slug
        return self._custom_endpoint_slug

    def _get_tts_endpoint(self, custom_endpoint_slug: str | None) -> str:
        slug = self._resolve_custom_slug(custom_endpoint_slug)
        if slug:
            return ENDPOINTS["custom_tts"].format(slug=slug)
        return ENDPOINTS["tts"]

    def _get_status_endpoint(
        self, generation_id: str, custom_endpoint_slug: str | None
    ) -> tuple[str, dict[str, Any] | None]:
        slug = self._resolve_custom_slug(custom_endpoint_slug)
        if slug:
            endpoint = ENDPOINTS["custom_generation"].format(
                slug=slug, generation_id=generation_id
            )
            return endpoint, None
        return ENDPOINTS["tts"], {"generation_id": generation_id}

    
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
            data = {"detail": response.text or "Unknown error"}
        
        message = (
            data.get("detail")
            or data.get("error")
            or data.get("message")
            or f"API error: {response.status_code}"
        )
        
        success_codes = [200, 201] if accept_201 else [200]
        if response.status_code in success_codes:
            return data
        elif response.status_code == 401:
            raise AuthenticationError(
                message=message,
                status_code=response.status_code,
                response_data=data,
            )
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                message=message,
                status_code=response.status_code,
                response_data=data,
                retry_after=float(retry_after) if retry_after else None,
            )
        else:
            raise APIError(
                message=message,
                status_code=response.status_code,
                response_data=data,
            )
    
    def _request(
        self,
        method: str,
        endpoint: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
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
        Generate speech from text.
        
        This method initiates TTS generation and returns immediately with a
        generation ID. Use status() to check progress, or tts_and_wait() for
        convenience.
        
        Args:
            text: The text to convert to speech.
            voice_id: The Voice ID to use (e.g., "emma"). For compatibility,
                you may also pass `voice` in kwargs.
            model: The model to use: "base" or "advanced". Defaults to "base".
            speed: Playback speed (0.5 to 2.0). Defaults to 1.0.
            cfg_weight: (Advanced model only) CFG weight (1.0 to 5.0).
            exaggeration: (Advanced model only) Exaggeration (0.0 to 1.0).
            audio_format: Output format ("wav" or "mp3"). Defaults to "wav".
            custom_endpoint_slug: Optional custom endpoint slug to route this request.
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
            ...     voice_id="emma",
            ...     model="base",
            ...     speed=1.1,
            ... )
            >>> print(gen_id)
        
        """
        text = validate_text(text)
        voice_alias = kwargs.pop("voice", None)
        if voice_id and voice_alias and voice_id != voice_alias:
            raise ValidationError("Provide only one of voice_id or voice.")
        voice_id = voice_id or voice_alias
        if not voice_id:
            raise ValidationError("voice_id is required.")

        model_normalized = self._normalize_model(model)
        payload: dict[str, Any] = {
            "text": text,
            "voice_id": voice_id,
            "model": model_normalized,
            "speed": speed,
            "audio_format": audio_format or get_config().default_format,
        }

        if model_normalized == "advanced":
            if cfg_weight is not None:
                payload["cfg_weight"] = cfg_weight
            if exaggeration is not None:
                payload["exaggeration"] = exaggeration

        payload.update(kwargs)

        logger.info(f"Starting TTS generation for text: {text[:50]}...")
        endpoint = self._get_tts_endpoint(custom_endpoint_slug)
        response = self._request("POST", endpoint, json=payload)

        generation_id = response.get("generation_id") or response.get("id")
        if not generation_id:
            raise APIError("No generation ID in response", response_data=response)
        
        logger.info(f"TTS generation started: {generation_id}")
        return generation_id
    
    def status(
        self,
        generation_id: str,
        custom_endpoint_slug: str | None = None,
    ) -> dict[str, Any]:
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
        endpoint, params = self._get_status_endpoint(generation_id, custom_endpoint_slug)
        return self._request("GET", endpoint, params=params)
    
    def tts_and_wait(
        self,
        text: str,
        voice_id: str | None = None,
        model: Model = "base",
        speed: float = 1.0,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        cfg_weight: float | None = None,
        exaggeration: float | None = None,
        audio_format: AudioFormat | None = None,
        custom_endpoint_slug: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate speech and wait for completion.
        
        This is a convenience method that combines tts() and status() polling.
        
        Args:
            text: The text to convert to speech.
            voice_id: The Voice ID to use for generation.
            model: The model to use: "base" or "advanced".
            speed: Playback speed (0.5 to 2.0).
            poll_interval: Time between status checks in seconds.
            timeout: Maximum time to wait for completion in seconds.
            cfg_weight: (Advanced model only) CFG weight.
            exaggeration: (Advanced model only) Exaggeration level.
            audio_format: Output format ("wav" or "mp3").
            custom_endpoint_slug: Optional custom endpoint slug to route this request.
            **kwargs: Additional parameters passed to the API.
            
        Returns:
            The audio URL for the completed generation.
            
        Raises:
            TimeoutError: If generation doesn't complete within timeout.
            GenerationError: If generation fails.
            
        Example:
            >>> audio_url = client.tts_and_wait(
            ...     "Hello, welcome to Audixa AI!",
            ...     voice_id="emma",
            ...     timeout=120,
            ... )
            >>> print(audio_url)
        """
        generation_id = self.tts(
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
        
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Generation timed out after {timeout}s",
                    timeout_seconds=timeout,
                )
            
            status_response = self.status(generation_id, custom_endpoint_slug=custom_endpoint_slug)
            status = str(status_response.get("status", "unknown")).upper()
            
            logger.debug(f"Generation {generation_id} status: {status}")
            
            if status == "COMPLETED":
                audio_url = status_response.get("url") or status_response.get("audio_url")
                if not audio_url:
                    raise GenerationError(
                        "Generation completed but no audio URL provided",
                        generation_id=generation_id,
                        response_data=status_response,
                    )
                logger.info(f"Generation completed: {audio_url}")
                return audio_url
            
            elif status in {"FAILED", "EXPIRED"}:
                error_msg = (
                    status_response.get("error_message")
                    or status_response.get("detail")
                    or status_response.get("error")
                    or "Unknown error"
                )
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
        voice_id: str | None = None,
        model: Model = "base",
        speed: float = 1.0,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        cfg_weight: float | None = None,
        exaggeration: float | None = None,
        audio_format: AudioFormat | None = None,
        custom_endpoint_slug: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate speech and save to a file.
        
        This combines tts_and_wait() with audio download.
        
        Args:
            text: The text to convert to speech.
            filepath: Output file path.
            voice_id: The Voice ID to use for generation.
            model: The model to use: "base" or "advanced".
            speed: Playback speed (0.5 to 2.0).
            poll_interval: Time between status checks in seconds.
            timeout: Maximum time to wait for completion in seconds.
            cfg_weight: (Advanced model only) CFG weight.
            exaggeration: (Advanced model only) Exaggeration.
            audio_format: Output format ("wav" or "mp3").
            custom_endpoint_slug: Optional custom endpoint slug to route this request.
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
            ...     voice_id="emma",
            ... )
            >>> print(f"Audio saved to: {filepath}")
        """
        # Validate output path
        output_path = validate_output_filepath(filepath, format_name=audio_format)
        
        # Generate and wait for audio URL
        audio_url = self.tts_and_wait(
            text,
            voice_id=voice_id,
            model=model,
            speed=speed,
            poll_interval=poll_interval,
            timeout=timeout,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
            audio_format=audio_format,
            custom_endpoint_slug=custom_endpoint_slug,
            **kwargs,
        )
        
        # Download to file
        download_file_sync(audio_url, output_path, timeout=self._timeout)
        
        return str(output_path)
    
    def list_voices(
        self,
        model: Model | None = None,
        limit: int = 100,
        offset: int = 0,
        include_metadata: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """
        List available voices for a specific model.
        
        Args:
            model: Optional model filter: "base" or "advanced".
            limit: Maximum number of results to return (1-500).
            offset: Number of results to skip.
            include_metadata: If True, return the full response with pagination.
        
        Returns:
            List of voice dictionaries, each containing:
            - voice_id: Voice ID for use in tts()
            - name: Display name
            - model: Compatible model ("base" or "advanced")
            - gender: Voice gender (e.g., "Male", "Female")
            - accent: Voice accent (e.g., "American", "British")
            - free: Whether the voice is free tier
            - is_custom: Whether this is a user-created voice
            - description: Voice description
            
        Example:
            >>> voices = client.list_voices(model="base")
            >>> for voice in voices:
            ...     print(f"{voice['voice_id']}: {voice['name']}")
        
        See Also:
            Audixa documentation: https://docs.audixa.ai/api/voices
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if model:
            params["model"] = self._normalize_model(model)
        logger.debug("Fetching available voices")
        response = self._request("GET", ENDPOINTS["voices"], params=params)
        if include_metadata:
            return response
        voices = response.get("voices", [])
        logger.info(f"Found {len(voices)} available voices")
        return voices

    def history(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
        include_metadata: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """
        Retrieve generation history.
        
        Args:
            limit: Maximum number of results to return (1-100).
            offset: Number of results to skip.
            status: Optional status filter (IN_QUEUE, GENERATING, COMPLETED, FAILED, EXPIRED).
            include_metadata: If True, return the full response with pagination.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        response = self._request("GET", ENDPOINTS["history"], params=params)
        if include_metadata:
            return response
        return response.get("history", [])

    
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
