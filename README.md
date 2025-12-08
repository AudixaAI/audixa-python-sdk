# Audixa Python SDK

[![PyPI version](https://badge.fury.io/py/audixa.svg)](https://badge.fury.io/py/audixa)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official Python SDK for the [Audixa AI](https://audixa.ai) Text-to-Speech API. Production-ready with both synchronous and asynchronous support.

## Features

- ✅ **Simple API** - One-line TTS generation
- ✅ **Sync & Async** - Full support for both paradigms
- ✅ **Type Hints** - Complete type annotations (Python 3.10+)
- ✅ **Retry Logic** - Automatic retries with exponential backoff
- ✅ **Rate Limiting** - Built-in concurrency control for async
- ✅ **File Downloads** - Download audio directly to disk
- ✅ **Error Handling** - Comprehensive exception hierarchy

## Installation

```bash
pip install audixa
```

For async file downloads, also install:
```bash
pip install aiofiles
```

## Quick Start

### Synchronous Usage

```python
import audixa

# Set your API key
audixa.set_api_key("your-api-key")
# Or use environment variable: AUDIXA_API_KEY

# Generate TTS and get audio URL
audio_url = audixa.tts_and_wait(
    "Hello, world! Welcome to Audixa AI text-to-speech.",
    voice="am_ethan",
)
print(f"Audio URL: {audio_url}")

# Or save directly to file
audixa.tts_to_file(
    "Hello, world! Welcome to Audixa.",
    "output.wav",
    voice="am_ethan",
)
```

### Asynchronous Usage

```python
import asyncio
import audixa

audixa.set_api_key("your-api-key")

async def main():
    # Generate TTS asynchronously
    audio_url = await audixa.atts_and_wait(
        "Hello, world! This is async generation.",
        voice="am_ethan",
    )
    print(f"Audio URL: {audio_url}")
    
    # Save to file
    await audixa.atts_to_file(
        "Hello from async world!",
        "output.wav",
        voice="am_ethan",
    )

asyncio.run(main())
```

### Using Models and Parameters

```python
import audixa

audixa.set_api_key("your-api-key")

# Base model with speed adjustment
audio_url = audixa.tts_and_wait(
    "Welcome to Audixa AI, your text-to-speech solution.",
    voice="am_ethan",
    model="base",
    speed=1.1,  # Slightly faster (0.5 to 2.0)
)

# Advance model with emotion
audio_url = audixa.tts_and_wait(
    "This is exciting news! We have launched.",
    voice="am_ethan",
    model="advance",
    emotion="happy",
    temperature=0.8,
    top_p=0.9,
    do_sample=True,
)
```

### Low-Level Client API

For more control, use the client classes directly:

```python
from audixa import AudixaClient

# Create client with custom settings
client = AudixaClient(
    api_key="your-api-key",
    timeout=60.0,
    max_retries=5,
)

# Use as context manager
with client:
    # Start generation (non-blocking)
    gen_id = client.tts(
        "Hello, world! Welcome to Audixa.",
        voice="am_ethan",
    )
    
    # Check status manually
    status = client.status(gen_id)
    print(f"Status: {status['status']}")
    
    # List available voices
    voices = client.list_voices()
    for voice in voices:
        print(f"{voice['id']}: {voice['name']}")
```

### Async Client

```python
from audixa import AsyncAudixaClient
import asyncio

async def main():
    async with AsyncAudixaClient(
        api_key="your-api-key",
        max_concurrency=10,  # Rate limit protection
    ) as client:
        # Concurrent generation
        texts = [
            "First sentence to generate.",
            "Second sentence to generate.",
            "Third sentence to generate.",
        ]
        tasks = [
            client.tts_and_wait(text, voice="am_ethan")
            for text in texts
        ]
        urls = await asyncio.gather(*tasks)
        print(urls)

asyncio.run(main())
```

## API Reference

### Configuration Functions

| Function | Description |
|----------|-------------|
| `set_api_key(key)` | Set the global API key |
| `set_base_url(url)` | Set the API base URL |

### TTS Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | str | Yes | Text to convert (min 30 chars) |
| `voice` | str | Yes | Voice ID (e.g., "am_ethan") |
| `model` | str | No | "base" or "advance" (default: "base") |
| `speed` | float | No | 0.5 to 2.0 (default: 1.0) |

#### Advance Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `emotion` | str | "neutral" | "neutral", "happy", "sad", "angry", "surprised" |
| `temperature` | float | 0.9 | Randomness control (0.7-0.9) |
| `top_p` | float | 0.9 | Nucleus sampling (0.7-0.98) |
| `do_sample` | bool | false | Enable sampling for varied outputs |

### Synchronous Functions

| Function | Description |
|----------|-------------|
| `tts(text, voice, ...)` | Start TTS generation, returns generation ID |
| `status(generation_id)` | Check generation status |
| `tts_and_wait(text, voice, timeout=300)` | Generate and wait, returns audio URL |
| `tts_to_file(text, filepath, voice)` | Generate and save to file |
| `list_voices()` | Get available voices |

### Asynchronous Functions

| Function | Description |
|----------|-------------|
| `atts(text, voice, ...)` | Async TTS generation |
| `astatus(generation_id)` | Async status check |
| `atts_and_wait(text, voice, timeout=300)` | Async generate and wait |
| `atts_to_file(text, filepath, voice)` | Async generate and save |
| `alist_voices()` | Async get voices |

## Error Handling

The SDK provides a comprehensive exception hierarchy:

```python
import audixa

try:
    audio_url = audixa.tts_and_wait(
        "Hello, this is a test message.",
        voice="am_ethan",
    )
except audixa.AuthenticationError:
    print("Invalid API key")
except audixa.RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}s")
except audixa.TimeoutError:
    print("Generation timed out")
except audixa.GenerationError as e:
    print(f"Generation failed: {e.message}")
except audixa.NetworkError:
    print("Network error occurred")
except audixa.AudixaError as e:
    print(f"General error: {e}")
```

### Exception Reference

| Exception | Description |
|-----------|-------------|
| `AudixaError` | Base exception for all SDK errors |
| `AuthenticationError` | Invalid or missing API key |
| `RateLimitError` | Rate limit exceeded (429) |
| `APIError` | General API error (4xx/5xx) |
| `NetworkError` | Connection/network failure |
| `TimeoutError` | Request or generation timeout |
| `GenerationError` | TTS generation failed |
| `UnsupportedFormatError` | Invalid audio format |
| `ValidationError` | Invalid input parameters |

## Audio Format

> **Note:** Currently, Audixa only supports **WAV** audio output. The SDK is designed for easy extensibility when new formats are added.

```python
# Correct - WAV format
audixa.tts_to_file("Hello", "output.wav", voice="am_ethan")

# Error - MP3 not yet supported  
audixa.tts_to_file("Hello", "output.mp3", voice="am_ethan")  # Raises UnsupportedFormatError
```

## Logging

Enable debug logging to see SDK activity:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or configure the audixa logger specifically
logging.getLogger("audixa").setLevel(logging.DEBUG)
```

## Production Best Practices

### 1. Use Environment Variables

```bash
export AUDIXA_API_KEY="your-api-key"
```

```python
import audixa
# API key is automatically loaded from environment
audio_url = audixa.tts_and_wait("Hello!", voice="am_ethan")
```

### 2. Handle Errors Gracefully

```python
import audixa
import time

def generate_audio(text: str, voice: str) -> str | None:
    try:
        return audixa.tts_and_wait(text, voice=voice, timeout=120)
    except audixa.RateLimitError:
        time.sleep(60)
        return generate_audio(text, voice)  # Retry
    except audixa.AudixaError as e:
        logging.error(f"TTS failed: {e}")
        return None
```

### 3. Use Async for High Throughput

```python
async def batch_generate(texts: list[str], voice: str) -> list[str]:
    async with AsyncAudixaClient(max_concurrency=5) as client:
        tasks = [client.tts_and_wait(text, voice=voice) for text in texts]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

### 4. Configure Timeouts

```python
client = AudixaClient(
    api_key="your-key",
    timeout=60.0,      # HTTP request timeout
    max_retries=5,     # Retry attempts
)

audio_url = client.tts_and_wait(
    "Long text...",
    voice="am_ethan",
    timeout=300.0,     # Wait timeout for generation
)
```

## Building from Source

```bash
# Clone the repository
git clone https://github.com/audixa/audixa-python.git
cd audixa-python

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Build package
python -m build
```

## Publishing to PyPI

```bash
# Build distribution
python -m build

# Upload to PyPI
twine upload dist/*
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Links

- 📖 [Documentation](https://docs.audixa.ai)
- 🎤 [Voice Library](https://docs.audixa.ai/voices)
- 🤖 [Models Guide](https://docs.audixa.ai/models)
- 🐛 [Issue Tracker](https://github.com/audixa/audixa-python/issues)
# audixa-python-sdk
