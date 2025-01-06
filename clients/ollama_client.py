import asyncio
import json
from typing import Any, Dict, Optional

import aiohttp


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize Ollama client with base URL"""
        self.base_url = base_url
        self.generate_endpoint = f"{base_url}/api/generate"

    async def generate(
        self,
        prompt: str,
        model: str = "llama3.2:1b",
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate response from Ollama API

        Args:
            prompt: The user prompt
            model: Model name (default: llama3.2:1b)
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            str: Complete generated response
        """
        payload = {
            "model": model,
            "prompt": prompt,
        }

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
                        raise Exception(f"Ollama API error: {error_text}")

                    # Handle streaming response
                    async for line in response.content:
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                full_response.append(data["response"])
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue

            return "".join(full_response)

        except Exception as e:
            raise Exception(f"Failed to generate response: {str(e)}")


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


# Example usage
if __name__ == "__main__":
    # Test the client
    try:
        response = sync_generate(
            prompt="Why is the sky blue?", system_prompt="You are a helpful assistant."
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {str(e)}")
