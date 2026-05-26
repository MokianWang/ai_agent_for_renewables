"""LLM client supporting DeepSeek API (OpenAI-compatible)."""

import os
import re
from openai import OpenAI


def _load_env():
    """Load .env file from project root."""
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".env",
    )
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if key and value and key not in os.environ:
                        os.environ[key] = value


_load_env()


def get_client():
    """Create and return an LLM client based on .env configuration."""
    provider = os.environ.get("LLM_PROVIDER", "deepseek")

    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        return OpenAI(api_key=api_key, base_url=base_url)
    elif provider == "openai":
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    elif provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def chat(client, messages, model=None):
    """Send a chat completion request. Returns the assistant's response text."""
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model = model or os.environ.get("LLM_MODEL", "deepseek-chat")

    if provider in ("deepseek", "openai"):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )
        return response.choices[0].message.content
    elif provider == "anthropic":
        system = None
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_messages.append(m)
        kwargs = {"model": model, "messages": user_messages, "max_tokens": 4096}
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text


def chat_stream(client, messages, model=None):
    """Stream chat completion tokens. Yields text chunks as they arrive."""
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model = model or os.environ.get("LLM_MODEL", "deepseek-chat")

    if provider in ("deepseek", "openai"):
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    elif provider == "anthropic":
        # Anthropic streaming
        system = None
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_messages.append(m)
        kwargs = {"model": model, "messages": user_messages, "max_tokens": 4096}
        if system:
            kwargs["system"] = system
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text


def parse_action(response_text: str):
    """Parse Thought/Action/Final Answer from LLM response."""
    thought = None
    action = None
    action_args = None
    final_answer = None

    thought_match = re.search(r"Thought:\s*(.+?)(?=\n(?:Action:|Final Answer:)|$)", response_text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", response_text)
    if action_match:
        action = action_match.group(1).strip()
        raw_args = action_match.group(2).strip()
        action_args = _parse_args(raw_args)

    final_match = re.search(r"Final Answer:\s*(.+)", response_text, re.DOTALL)
    if final_match:
        final_answer = final_match.group(1).strip()

    return thought, action, action_args, final_answer


def _parse_args(args_str: str) -> dict:
    """Parse tool arguments from a string like 'prediction_type=\"load_power\", zone=1'."""
    if not args_str:
        return {}
    result = {}
    for part in args_str.split(","):
        part = part.strip()
        if "=" in part:
            key, _, value = part.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            try:
                value = int(value)
            except (ValueError, TypeError):
                pass
            result[key] = value
    return result
