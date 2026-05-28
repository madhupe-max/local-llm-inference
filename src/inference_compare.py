import argparse
import os
import time
from typing import Callable, Dict, Optional

import requests


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OPENAI_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def call_ollama(prompt: str, model: str = DEFAULT_OLLAMA_MODEL, base_url: str = DEFAULT_OLLAMA_URL) -> str:
    response = requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("response", "")


def call_openai(
    prompt: str,
    model: str = DEFAULT_OPENAI_MODEL,
    base_url: str = DEFAULT_OPENAI_URL,
    api_key: Optional[str] = None,
) -> str:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY is required for cloud inference")

    # Normalize to string to avoid unsupported operand errors in header composition.
    api_key = str(api_key).strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for cloud inference")

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def compare_inference(
    prompt: str,
    local_client: Callable[[str], str],
    cloud_client: Callable[[str], str],
) -> Dict[str, object]:
    local_start = time.perf_counter()
    local_response = local_client(prompt)
    local_ms = (time.perf_counter() - local_start) * 1000

    cloud_start = time.perf_counter()
    cloud_response = cloud_client(prompt)
    cloud_ms = (time.perf_counter() - cloud_start) * 1000

    return {
        "prompt": prompt,
        "local_response": local_response,
        "cloud_response": cloud_response,
        "local_latency_ms": round(local_ms, 2),
        "cloud_latency_ms": round(cloud_ms, 2),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare local Ollama inference with cloud inference")
    parser.add_argument("--prompt", required=True, help="Prompt to send to both models")
    parser.add_argument("--local-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--cloud-model", default=DEFAULT_OPENAI_MODEL)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    results = compare_inference(
        prompt=args.prompt,
        local_client=lambda p: call_ollama(
            p,
            model=args.local_model,
            base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
        ),
        cloud_client=lambda p: call_openai(
            p,
            model=args.cloud_model,
            base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_URL),
        ),
    )

    print("Prompt:", results["prompt"])
    print("\nLocal (Ollama):")
    print(results["local_response"])
    print(f"Latency: {results['local_latency_ms']}ms")
    print("\nCloud (OpenAI):")
    print(results["cloud_response"])
    print(f"Latency: {results['cloud_latency_ms']}ms")


if __name__ == "__main__":
    main()
