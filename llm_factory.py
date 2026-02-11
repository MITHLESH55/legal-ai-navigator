"""
LLM Factory Module
Provides centralized LLM initialization supporting both Groq and Gemini.
Set LLM_PROVIDER env variable to 'groq' or 'gemini' to switch providers.
"""

import os
import logging
from typing import Literal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLMProvider = Literal["groq", "gemini"]


def get_llm(
    model_size: Literal["small", "large"] = "large",
    temperature: float = 0.0
):
    """
    Get an LLM instance based on the LLM_PROVIDER environment variable.
    
    Args:
        model_size: "small" for faster/cheaper models, "large" for powerful models
        temperature: Temperature setting for the LLM (0.0 = deterministic, 1.0 = creative)
    
    Returns:
        LLM instance (ChatGroq or ChatGoogleGenerativeAI)
    
    Environment Variables:
        LLM_PROVIDER: "groq" or "gemini" (default: "groq")
        GROQ_API_KEY: Required if provider is "groq"
        GEMINI_API_KEY: Required if provider is "gemini"
    """

    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    # =========================
    # GROQ (DEFAULT / STABLE)
    # =========================
    if provider == "groq":
        from langchain_groq import ChatGroq

        if model_size == "large":
            model = "llama-3.3-70b-versatile"
        else:
            model = "llama-3.1-8b-instant"

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is required when LLM_PROVIDER=groq"
            )

        logger.info(f"Initializing Groq LLM with model: {model}")

        return ChatGroq(
            model=model,
            api_key=api_key,
            temperature=temperature
        )

    # =========================
    # GEMINI (FIXED MODELS)
    # =========================
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # ✅ VALID, PUBLIC GEMINI MODELS
        if model_size == "large":
            model = "gemini-1.5-pro"
        else:
            model = "models/gemini-pro"

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required when LLM_PROVIDER=gemini"
            )

        logger.info(f"Initializing Gemini LLM with model: {model}")

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {provider}. Must be 'groq' or 'gemini'"
        )


def get_provider() -> str:
    """Get the current LLM provider."""
    return os.getenv("LLM_PROVIDER", "groq").lower()
