import asyncio
import json
import os
from typing import Any, Dict, Optional

import aiohttp


class OllamaClient:
    """
    Client for interacting with Ollama API running in Docker

    Usage with Docker:
        - Default URL assumes Ollama running in Docker with port 11434 exposed
        - Can override URL via OLLAMA_HOST environment variable
    """

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize Ollama client

        Args:
            base_url: Optional override for API URL.
                     Defaults to env var OLLAMA_HOST or http://localhost:11434
        """
        self.base_url = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.generate_endpoint = f"{self.base_url}/api/generate"
        print(f"Initialized Ollama client with URL: {self.base_url}")

    async def generate(
        self,
        prompt: str,
        model: str = "llama3.2:1b",
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate response from Ollama API

        Args:
            prompt: The user prompt
            model: Model name (default: llama3.2:1b)
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (default: 0.7)

        Returns:
            str: Complete generated response

        Raises:
            Exception: If API call fails or response processing fails
        """
        payload = {"model": model, "prompt": prompt, "temperature": temperature}

        if system_prompt:
            payload["system"] = system_prompt
        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            full_response = []
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.generate_endpoint, json=payload, timeout=30
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(
                            f"Ollama API error ({response.status}): {error_text}"
                        )

                    # Handle streaming response
                    async for line in response.content:
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if "error" in data:
                                raise Exception(f"Ollama API error: {data['error']}")
                            if "response" in data:
                                full_response.append(data["response"])
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to decode line: {line}")
                            continue

            return "".join(full_response)

        except Exception as e:
            raise Exception(f"Failed to generate response: {str(e)}")

    async def check_model_availability(self, model: str = "llama3.2:1b") -> bool:
        """
        Check if the specified model is available

        Args:
            model: Model name to check

        Returns:
            bool: True if model is available, False otherwise
        """
        try:
            response = await self.generate("test", model=model, max_tokens=1)
            return True
        except Exception:
            return False


# Helper function for synchronous calls
def sync_generate(
    prompt: str,
    model: str = "llama3.2:1b",
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Synchronous wrapper for generate method

    Returns:
        str: Generated response text
    """

    async def _generate():
        client = OllamaClient()
        return await client.generate(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )

    return asyncio.run(_generate())


if __name__ == "__main__":
    # Test the client
    async def test_client():
        client = OllamaClient()
        print(f"Testing connection to Ollama at {client.base_url}")

        try:
            # Check model availability
            model = "llama3.2:1b"
            is_available = await client.check_model_availability(model)
            if not is_available:
                print(f"Warning: Model {model} may not be available")
                return

            # Test generation
            response = await client.generate(
                prompt="What can you tell me about autonomous agents?",
                system_prompt="You are a helpful AI assistant.",
                max_tokens=100,
            )
            print(f"\nTest response:\n{response}")

        except Exception as e:
            print(f"Error during test: {str(e)}")

    asyncio.run(test_client())
