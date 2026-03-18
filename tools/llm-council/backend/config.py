"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv, find_dotenv

# Automatically walk up the tree to find the .env file in the main project directory
load_dotenv(find_dotenv(usecwd=True))

# Note: OPENAI_API_KEY, ANTHROPIC_API_KEY, and GEMINI_API_KEY are loaded 
# automatically from the .env file by the respective clients in openrouter.py.

# Council members - list of model identifiers with 'provider/' prefix
COUNCIL_MODELS = [
    "anthropic/claude-opus-4.6",     # The Lead Architect (Code & Context)
    "openai/gpt-5.4-thinking",       # The Adversarial Auditor (Finding Leakage)
    "google/gemini-3.1-pro",         # The Data Pragmatist (Logic & Multimodal)
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "anthropic/claude-opus-4.6"

# Data directory for conversation storage
DATA_DIR = "data/conversations"