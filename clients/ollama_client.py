import asyncio
import json
import os
from typing import Any, Dict, Optional

import aiohttp


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self, model_name: str = "llama3.2:1b"):
        """Initialize Ollama client"""
        self.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model_name = model_name
        print(
            f"Initialized Ollama client with URL: {self.base_url} and model: {self.model_name}"
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate response from Ollama API"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
        }

        if system_prompt:
            payload["system"] = system_prompt
        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate", json=payload, timeout=30
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(
                            f"Ollama API error ({response.status}): {error_text}"
                        )

                    data = await response.json()

                    if "error" in data:
                        raise Exception(f"Ollama API error: {data['error']}")

                    return data.get("response", "")

        except Exception as e:
            raise Exception(f"Failed to generate response: {str(e)}")

    async def check_model_availability(self) -> bool:
        """Check if the configured model is available"""
        try:
            response = await self.generate("test", max_tokens=1)
            return True
        except Exception:
            return False


# Helper function for synchronous calls
def sync_generate(
    prompt: str,
    model_name: str = "llama3.2:1b",
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Synchronous wrapper for generate method"""

    async def _generate():
        client = OllamaClient(model_name)
        return await client.generate(
            prompt=prompt,
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
            is_available = await client.check_model_availability()
            if not is_available:
                print(f"Warning: Model {client.model_name} may not be available")
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
