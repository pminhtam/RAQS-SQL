

"""
Source https://github.com/ShayanTalaei/CHESS/blob/main/src/llm/engine_configs.py
Chứa config tới các api llm khác nhau


"""


import os
from typing import Dict, Any
from langchain_groq import ChatGroq

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI

"""
This module defines configurations for various language models using the langchain library.
Each configuration includes a constructor, parameters, and an optional preprocessing function.
"""

ENGINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gemini-pro": {
        "constructor": ChatGoogleGenerativeAI,
        "params": {"model": "gemini-pro", "temperature": 0, "convert_system_message_to_human": True},
        "preprocess": lambda x: x.to_messages()
    },
    "gemini-1.5-pro-latest": {
        "constructor": ChatGoogleGenerativeAI,
        "params": {"model": "gemini-1.5-pro-latest", "temperature": 0, "convert_system_message_to_human": True},
        "preprocess": lambda x: x.to_messages()
    },
    "gemini-1.5-flash": {
        "constructor": ChatGoogleGenerativeAI,
        "params": {"model": "gemini-1.5-flash-latest", "temperature": 0, "convert_system_message_to_human": True},
        "preprocess": lambda x: x.to_messages()
    },
    "gemini-1.5-flash-8B": {
        "constructor": ChatGoogleGenerativeAI,
        "params": {"model": "gemini-1.5-flash-8b", "temperature": 0, "convert_system_message_to_human": True},
        "preprocess": lambda x: x.to_messages()
    },
    "gpt-4.1-mini": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4.1-mini", "temperature": 0}
    },"gpt-4.1": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4.1", "temperature": 0}
    },"gpt-4o-mini": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4o-mini", "temperature": 0}
    },
    "gpt-4o": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4o", "temperature": 0}
    },
    "claude-3-5-sonnet-20241022": {
        "constructor": ChatAnthropic,
        "params": {"model": "claude-3-5-sonnet-20241022", "temperature": 0}
    },
    "claude-3-5-sonnet-20240620": {
            "constructor": ChatAnthropic,
            "params": {"model": "claude-3-5-sonnet-20240620", "temperature": 0}
        },
    "claude-3-5-haiku": {
            "constructor": ChatAnthropic,
            "params": {"model": "claude-3-5-haiku", "temperature": 0}
        },
    "claude-3-5-opus": {
            "constructor": ChatAnthropic,
            "params": {"model": "claude-3-5-opus", "temperature": 0}
        },
    "claude-3-sonnet": {
                "constructor": ChatAnthropic,
                "params": {"model": "claude-3-sonnet", "temperature": 0}
            },
    "claude-3-haiku": {
                "constructor": ChatAnthropic,
                "params": {"model": "claude-3-haiku", "temperature": 0}
            },
    "groq-llama3-70b-8192": {
                "constructor": ChatGroq,
                "params": {"model": "llama3-70b-8192", "temperature": 0}
            },
    "groq-llama-3.3-70b-specdec": {
                "constructor": ChatGroq,
                "params": {"model": "llama-3.3-70b-specdec", "temperature": 0}
            },
    "groq-llama-3.3-70b-versatile": {
                "constructor": ChatGroq,
                "params": {"model": "llama-3.3-70b-versatile", "temperature": 0}
            },
    "groq-llama-3.1-8b-instant": {
                "constructor": ChatGroq,
                "params": {"model": "llama-3.1-8b-instant", "temperature": 0}
            },
    "groq-deepseek-r1-distill-llama-70b": {
                "constructor": ChatGroq,
                "params": {"model": "deepseek-r1-distill-llama-70b", "temperature": 0}
            },
    "mistral-large-latest": {
                "constructor": ChatMistralAI,
                "params": {"model": "mistral-large-latest", "temperature": 0}
            },
    "mistral-large-2407": {
                "constructor": ChatMistralAI,
                "params": {"model": "mistral-large-2407", "temperature": 0}
            },
    "mistral-small-latest": {
                "constructor": ChatMistralAI,
                "params": {"model": "mistral-small-latest", "temperature": 0}
            },
    "open-mixtral-8x22b": {
                "constructor": ChatMistralAI,
                "params": {"model": "open-mixtral-8x22b", "temperature": 0}
            },

    "finetuned_nl2sql": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "AI4DS/NL2SQL_DeepSeek_33B",
            "openai_api_key": "EMPTY",
            "openai_api_base": "/v1",
            "max_tokens": 400,
            "temperature": 0,
            "model_kwargs": {
                "stop": ["```\n", ";"]
            }
        }
    },
    "meta-llama/Meta-Llama-3-70B-Instruct": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "meta-llama/Meta-Llama-3-70B-Instruct",
            "openai_api_key": "EMPTY",
            "openai_api_base": "/v1",
            "max_tokens": 600,
            "temperature": 0,
            "model_kwargs": {
                "stop": [""]
            }
        }
    },
    "qwen/qwen2.5-coder-32b-instruct":{
        "constructor": ChatOpenAI,
        "params": {
            "model": "qwen/qwen2.5-coder-32b-instruct",
            "openai_api_key": os.environ["NVIDIA_API_KEY"],
            "openai_api_base": "https://integrate.api.nvidia.com/v1"
        }
    },
    "meta/llama-3.3-70b-instruct":{
        "constructor": ChatOpenAI,
        "params": {
            "model": "meta/llama-3.3-70b-instruct",
            "openai_api_key": os.environ["NVIDIA_API_KEY"],
            "openai_api_base": "https://integrate.api.nvidia.com/v1"
        }
    },
    "deepseek-ai/deepseek-r1":{
        "constructor": ChatOpenAI,
        "params": {
            "model": "deepseek-ai/deepseek-r1",
            "openai_api_key": os.environ["NVIDIA_API_KEY"],
            "openai_api_base": "https://integrate.api.nvidia.com/v1"
        }
    },
    "meta/llama-3.1-405b-instruct":{
        "constructor": ChatOpenAI,
        "params": {
            "model": "meta/llama-3.1-405b-instruct",
            "openai_api_key": os.environ["NVIDIA_API_KEY"],
            "openai_api_base": "https://integrate.api.nvidia.com/v1"
        }
    },
    "google/gemma-2-27b-it":{
        "constructor": ChatOpenAI,
        "params": {
            "model": "google/gemma-2-27b-it",
            "openai_api_key": os.environ["NVIDIA_API_KEY"],
            "openai_api_base": "https://integrate.api.nvidia.com/v1"
        }
    },
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free":{
        "constructor": ChatOpenAI,
        "params": {
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "openai_api_key": os.environ["TOGETHER_API_KEY"],
            "max_tokens": 400,
            "openai_api_base": "https://api.together.xyz/v1"
        }
    },
}
