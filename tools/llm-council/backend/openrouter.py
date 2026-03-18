"""Direct API client for making LLM requests to OpenAI, Anthropic, and Google."""

import os
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv, find_dotenv

# Automatically walk up the tree to find the .env file in the main project directory
load_dotenv(find_dotenv(usecwd=True))

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from google import genai
from google.genai import types

async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via direct Provider APIs.

    Args:
        model: Model identifier with provider prefix (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    try:
        # Parse the provider prefix
        if "/" in model:
            provider, model_name = model.split("/", 1)
        else:
            provider = "openai"
            model_name = model

        content = None
        reasoning = None

        if provider == "openai":
            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                timeout=timeout
            )
            message = response.choices[0].message
            content = message.content
            # Handle o1/o3 reasoning details if present in the response
            if hasattr(message, "reasoning_details"):
                reasoning = message.reasoning_details

        elif provider == "anthropic":
            client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            
            # Anthropic requires extracting the system prompt from the messages array
            system_prompt = ""
            anthropic_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    anthropic_messages.append(msg)

            kwargs = {
                "model": model_name,
                "messages": anthropic_messages,
                "max_tokens": 8192,
                "timeout": timeout
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = await client.messages.create(**kwargs)
            content = response.content[0].text

        elif provider == "google":
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            
            # Convert standard messages to Google's genai.types format
            formatted_contents = []
            system_instruction = None
            
            for msg in messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                elif msg["role"] == "user":
                    formatted_contents.append(
                        types.Content(role="user", parts=[types.Part.from_text(text=msg["content"])])
                    )
                elif msg["role"] == "assistant":
                    formatted_contents.append(
                        types.Content(role="model", parts=[types.Part.from_text(text=msg["content"])])
                    )

            config = types.GenerateContentConfig()
            if system_instruction:
                config.system_instruction = system_instruction

            response = await client.aio.models.generate_content(
                model=model_name,
                contents=formatted_contents,
                config=config
            )
            content = response.text

        return {
            'content': content,
            'reasoning_details': reasoning
        }

    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None

async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    # Create tasks for all models
    tasks = [query_model(model, messages) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}