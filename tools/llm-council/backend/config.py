"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv, find_dotenv

# Automatically walk up the tree to find the .env file in the main project directory
load_dotenv(find_dotenv(usecwd=True))

# Note: OPENAI_API_KEY, ANTHROPIC_API_KEY, and GEMINI_API_KEY are loaded 
# automatically from the .env file by the respective clients in openrouter.py.

# Council members - list of model identifiers with 'provider/' prefix
# Council members - list of model identifiers with 'provider/' prefix
COUNCIL_MODELS = [
    "anthropic/claude-3-opus-20240229",  # Or "anthropic/claude-3-7-sonnet-20250219" for the newer architecture
    "openai/o3-mini",                    # OpenAI's actual "thinking" model identifier
    "google/gemini-2.5-pro",             # Or gemini-1.5-pro depending on your Google tier
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "anthropic/claude-3-opus-20240229"
# Data directory for conversation storage
DATA_DIR = "data/conversations"