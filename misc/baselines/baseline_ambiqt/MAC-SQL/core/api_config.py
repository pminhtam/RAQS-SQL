import os
# set your OPENAI_API_BASE, OPENAI_API_KEY here!
# OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "your_own_api_base")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_own_api_key")
# api_key = os.getenv("TOGETHER_API_KEY")
# api_key = os.getenv("INFERENCE_API_KEY")
from openai import OpenAI

MODEL_NAME = 'gpt-4.1' #
# MODEL_NAME = 'qwen/qwen2.5-coder-32b-instruct' # nvidia
# MODEL_NAME = "meta/llama-3.3-70b-instruct"   # nvidia
# MODEL_NAME = 'CodeLlama-7b-hf'
# MODEL_NAME = 'gpt-4-32k' # 0613版本
# MODEL_NAME = 'gpt-4' # 0613版本
# MODEL_NAME = 'gpt-35-turbo-16k' # 0613版本

if "gpt" in MODEL_NAME:
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
elif MODEL_NAME == "deepseek-ai/deepseek-r1" or MODEL_NAME == "qwen/qwen2.5-coder-32b-instruct" or MODEL_NAME == "meta/llama-3.3-70b-instruct":
    api_key = os.getenv("NVIDIA_API_KEY")
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )
elif "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" ==  model_llm_api or model_llm_api == "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free":
    api_key = os.environ["TOGETHER_API_KEY"]
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.together.xyz/v1",
    )
elif "deepseek/deepseek-r1" in model_llm_api or "meta-llama/llama-3.3-70b-instruct" in model_llm_api:
    api_key = os.environ["INFERENCE_API_KEY"]
    client = OpenAI(
        base_url="https://api.inference.net/v1",
        api_key=api_key
    )