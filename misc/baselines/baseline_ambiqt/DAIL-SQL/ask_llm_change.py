import argparse
import os
import json
import re

from openai import OpenAI
from tqdm import tqdm

from tenacity import (
  retry,
  stop_after_attempt,
  wait_random_exponential,
)  # for exponential backoff

QUESTION_FILE = "questions.json"

def openai_completion(prompt,max_tokens=1000):
    response = openai.chat.completions.create(
        model=model_llm_api,
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ],
        max_tokens=max_tokens,
        temperature=0
    )
    prompt_token = response.usage.prompt_tokens
    response_token = response.usage.completion_tokens
    return response.choices[0].message.content, prompt_token, response_token

@retry(wait=wait_random_exponential(min=10, max=1000), stop=stop_after_attempt(20))
def llm_api_completion(prompt):
    if "gpt" in model_llm_api:
        response, prompt_token, response_token = openai_completion(prompt,max_tokens=1000)
    elif "deepseek-ai/deepseek-r1" in model_llm_api  or "qwen/qwen2.5-coder-32b-instruct" == model_llm_api or "meta/llama-3.3-70b-instruct" == model_llm_api:
        response, prompt_token, response_token = openai_completion(prompt,max_tokens=10000)
        # time.sleep(10)
    elif "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" ==  model_llm_api  or model_llm_api == "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free":
        # print("Using together.xyz")
        response, prompt_token, response_token = openai_completion(prompt,max_tokens=1000)
    elif "deepseek/deepseek-r1-0528" in model_llm_api.lower():
        response, prompt_token, response_token = openai_completion(prompt,max_tokens=10000)
    elif "deepseek/deepseek-r1" in model_llm_api or "meta-llama/llama-3.3-70b-instruct/fp-16" == model_llm_api:
        print("Using inference.net")
        response, prompt_token, response_token = openai_completion(prompt,max_tokens=10000)
    return response, prompt_token, response_token
def extract_sql_queries(text):

    # if "```sql" in text:
    #     text = text.split("```sql")[1]

    sql_pattern = re.compile(r"(SELECT.*?[;]|SELECT.*?[]]|SELECT.*?[`])", re.DOTALL | re.IGNORECASE)

    matches = sql_pattern.findall(text)

    queries = [m.strip('` \n').replace("]",";") for m in matches] # only keep sufficiently long matches
    return queries
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_path", type=str)
    parser.add_argument("--model", type=str)

    args = parser.parse_args()
    model_llm_api = "gpt-4.1"
    # model_llm_api = "deepseek-ai/deepseek-r1" # nvidia
    # model_llm_api = "qwen/qwen2.5-coder-32b-instruct"   # nvidia
    # model_llm_api = "meta/llama-3.3-70b-instruct"   # nvidia
    # model_llm_api = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"  # TOGETHER_API_KEY
    # model_llm_api = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
    # model_llm_api = "deepseek/deepseek-r1-0528:free"
    # model_llm_api = "deepseek/deepseek-r1/fp-8"   # inference.net
    # model_llm_api = "meta-llama/llama-3.3-70b-instruct/fp-16"   # inference.net
    # model_llm_api = args.model
    if "gpt" in model_llm_api:
        api_key = os.environ["OPENAI_API_KEY"]
        openai = OpenAI(api_key=api_key)
    elif "deepseek-ai/deepseek-r1" == model_llm_api  or "qwen/qwen2.5-coder-32b-instruct" == model_llm_api or "meta/llama-3.3-70b-instruct" == model_llm_api:
        api_key = os.environ["NVIDIA_API_KEY"]
        openai = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
    elif "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" == model_llm_api or model_llm_api == "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free":
        api_key = os.environ["TOGETHER_API_KEY"]
        openai = OpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
        )
    elif "deepseek/deepseek-r1-0528" in model_llm_api:
        api_key = os.environ["OPENROUTER_API_KEY"]
        openai = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
    elif "deepseek/deepseek-r1" in model_llm_api or "meta-llama/llama-3.3-70b-instruct/fp-16" == model_llm_api:
        api_key = os.environ["INFERENCE_API_KEY"]
        openai = OpenAI(
            base_url="https://api.inference.net/v1",
            api_key=api_key
        )
    # args.question = "dataset/process_col/SPIDER-TEST_TEXT_9-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-4096"
    # output_path = "col_output_" + model_llm_api.replace("/", "_") + ".json"
    args.question = "dataset/process_tbl/SPIDER-TEST_TEXT_9-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-4096"
    output_path = "tbl_output_" + model_llm_api.replace("/", "_") + ".json"
    all_input_tokens = 0
    all_output_tokens = 0
    output_path_bk = output_path.replace(".json", "_bk.jsonl")
    f_bk = open(output_path_bk, 'w')
    questions_json = json.load(open(os.path.join(args.question, QUESTION_FILE), "r"))
    # questions = [_["prompt"] for _ in questions_json["questions"]]
    for ques_item in tqdm(questions_json["questions"]):
        prompt = ques_item["prompt"]
        response, prompt_token, response_token = llm_api_completion(prompt)
        print("Response: ", response)
        queries = extract_sql_queries(response)
        print("Queries: ", queries, " prompt_token: ", prompt_token, " response_token: ", response_token)
        ques_item["queries"] = queries
        ques_item["response_ori"] = response
        ques_item["prompt_token"] = prompt_token
        ques_item["response_token"] = response_token
        all_input_tokens += prompt_token
        all_output_tokens += response_token
        # import pdb; pdb.set_trace()
        f_bk.write(json.dumps(ques_item, ensure_ascii=False) + "\n")
    json.dump(questions_json["questions"], open(output_path, 'w'), indent=4, ensure_ascii=False)
    print(f"Total input tokens: {all_input_tokens}")
    print(f"Total output tokens: {all_output_tokens}")
"""
python ask_llm_change.py
"""