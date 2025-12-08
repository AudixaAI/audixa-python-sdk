"""
Audixa SDK Exception Classes.

Comprehensive exception hierarchy for handling all error scenarios
in the Audixa API client.
"""

from __future__ import annotations

from typing import Any


class AudixaError(Exception):
    """
    Base exception for all Audixa SDK errors.
    
    All other exceptions in this module inherit from this class,
    allowing you to catch all Audixa-related errors with a single except block.
    
    Attributes:
        message: Human-readable error description.
        status_code: HTTP status code if applicable.
        response_data: Raw API response data if available.
    
    Example:
        >>> try:
        ...     audixa.tts("Hello")
        ... except audixa.AudixaError as e:
        ...     print(f"Audixa error: {e}")
    """
    
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}
    
    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class AuthenticationError(AudixaError):
    """
    Raised when API authentication fails.
    
    This typically occurs when:
    - No API key is provided
    - The API key is invalid or expired
    - The API key doesn't have required permissions
    
    Example:
        >>> try:
        ...     audixa.tts("Hello")  # No API key set
        ... except audixa.AuthenticationError:
        ...     print("Please set your API key with audixa.set_api_key()")
    """
    
    def __init__(
        self,
        message: str = "Authentication failed. Please check your API key.",
        status_code: int | None = 401,
        response_data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status_code, response_data)


class RateLimitError(AudixaError):
    """
    Raised when the API rate limit is exceeded.
    
    The Audixa API enforces rate limits to ensure fair usage.
    When this error is raised, you should wait before retrying.
    
    Attributes:
        retry_after: Seconds to wait before retrying (if provided by API).
    
    Example:
        >>> try:
        ...     audixa.tts("Hello")
        ... except audixa.RateLimitError as e:
        ...     time.sleep(e.retry_after or 60)
    """
    
    def __init__(
        self,
        message: str = "Rate limit exceeded. Please wait before retrying.",
        status_code: int | None = 429,
        response_data: dict[str, Any] | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, status_code, response_data)
        self.retry_after = retry_after


class APIError(AudixaError):
    """
    Raised for general API errors (4xx/5xx responses).
    
    This is a catch-all for API errors that don't fit into
    more specific categories like AuthenticationError or RateLimitError.
    
    Example:
        >>> try:
        ...     audixa.tts("Hello")
        ... except audixa.APIError as e:
        ...     print(f"API error {e.status_code}: {e.message}")
    """
    pass


class NetworkError(AudixaError):
    """
    Raised when a network-level error occurs.
    
    This includes:
    - Connection timeouts
    - DNS resolution failures
    - Connection refused
    - SSL/TLS errors
    
    Example:
        >>> try:
        ...     audixa.tts("Hello")
        ... except audixa.NetworkError:
        ...     print("Network error. Please check your connection.")
    """
    
    def __init__(
        self,
        message: str = "Network error occurred. Please check your connection.",
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.original_error = original_error


class TimeoutError(AudixaError):
    """
    Raised when an operation times out.
    
    This can occur during:
    - HTTP request timeout
    - Polling timeout in tts_and_wait()
    - File download timeout
    
    Example:
        >>> try:
        ...     audixa.tts_and_wait("Hello", timeout=10)
        ... except audixa.TimeoutError:
        ...     print("Generation timed out")
    """
    
    def __init__(
        self,
        message: str = "Operation timed out.",
        timeout_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


class GenerationError(AudixaError):
    """
    Raised when TTS generation fails.
    
    This occurs when the API successfully receives the request
    but the generation process fails (e.g., invalid voice, text too long).
    
    Attributes:
        generation_id: The ID of the failed generation, if available.
    
    Example:
        >>> try:
        ...     audixa.tts_and_wait("Hello", voice="invalid")
        ... except audixa.GenerationError as e:
        ...     print(f"Generation {e.generation_id} failed: {e.message}")
    """
    
    def __init__(
        self,
        message: str = "TTS generation failed.",
        generation_id: str | None = None,
        status_code: int | None = None,
        response_data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status_code, response_data)
        self.generation_id = generation_id


class UnsupportedFormatError(AudixaError):
    """
    Raised when an unsupported audio format is requested.
    
    Currently, Audixa only supports WAV format. This error is raised
    when attempting to use other formats.
    
    Attributes:
        requested_format: The format that was requested.
        supported_formats: List of currently supported formats.
    
    Example:
        >>> try:
        ...     audixa.tts_to_file("Hello", "output.mp3")  # MP3 not supported
        ... except audixa.UnsupportedFormatError as e:
        ...     print(f"Use one of: {e.supported_formats}")
    """
    
    def __init__(
        self,
        requested_format: str,
        supported_formats: tuple[str, ...] | None = None,
    ) -> None:
        self.requested_format = requested_format
        self.supported_formats = supported_formats or ("wav",)
        message = (
            f"Unsupported audio format: '{requested_format}'. "
            f"Supported formats: {', '.join(self.supported_formats)}"
        )
        super().__init__(message)


class ValidationError(AudixaError):
    """
    Raised when input validation fails.
    
    This occurs before making an API request when the input
    parameters are invalid.
    
    Example:
        >>> try:
        ...     audixa.tts("")  # Empty text
        ... except audixa.ValidationError:
        ...     print("Text cannot be empty")
    """
    pass
