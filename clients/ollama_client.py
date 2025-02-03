import asyncio
import os
from typing import Optional

import aiohttp


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self, model_name: str = "llama3.2:1b-instruct-q3_K_M"):
        """Initialize Ollama client"""
        self.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model_name = model_name
        print(
            f"[DEBUG] Initialized Ollama client with URL: {self.base_url} and model: {self.model_name}"
        )

    async def check_model_availability(self) -> bool:
        """Check if the model is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.model_name, "prompt": "test", "stream": False},
                    timeout=5,
                ) as response:
                    if response.status == 200:
                        print(f"[DEBUG] Model {self.model_name} is available")
                        return True
                    else:
                        print(
                            f"[DEBUG] Model check failed with status {response.status}"
                        )
                        return False
        except Exception as e:
            print(f"[DEBUG] Model check error: {str(e)}")
            return False

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate response from Ollama API"""
        print(f"[DEBUG] Generating with prompt: {prompt[:100]}...")
        if system_prompt:
            print(f"[DEBUG] System prompt: {system_prompt[:100]}...")

        # Combine system prompt and user prompt if both are provided
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = prompt

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "temperature": temperature,
            "raw": False,  # Changed from True to False
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate", json=payload, timeout=30
                ) as response:
                    print(f"[DEBUG] Ollama response status: {response.status}")

                    if response.status != 200:
                        error_text = await response.text()
                        print(f"[DEBUG] Ollama error response: {error_text}")
                        raise Exception(
                            f"Ollama API error ({response.status}): {error_text}"
                        )

                    data = await response.json()
                    print(f"[DEBUG] Ollama response type: {type(data)}")
                    print(
                        f"[DEBUG] Ollama response keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}"
                    )

                    if "error" in data:
                        raise Exception(f"Ollama API error: {data['error']}")

                    response_text = data.get("response", "")
                    print(f"[DEBUG] Generated response length: {len(response_text)}")
                    return response_text

        except aiohttp.ClientError as e:
            print(f"[DEBUG] Network error: {str(e)}")
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            print(f"[DEBUG] Unexpected error: {str(e)}")
            raise


# Helper function for synchronous calls
def sync_generate(
    prompt: str,
    model_name: str = "llama3.2:1b-instruct-q3_K_M",
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
