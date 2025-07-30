

"""
Source https://github.com/ShayanTalaei/CHESS/blob/main/src/llm/prompts.py
Code tiền xử lý prompt trước khi gọi api llm
Các bước :
- Load template từ file
- Nhận các argument
- Tạo prompt từ template và argument

Step 1 trong quá trình gọi api

"""



import os
import logging
from typing import Any

from langchain.prompts import (
    PromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

TEMPLATES_ROOT_PATH = "./src/models/api_llm/templates"

def load_template(template_name: str) -> str:
    """
    Loads a template from a file.

    Args:
        template_name (str): The name of the template to load.

    Returns:
        str: The content of the template.
    """
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)

    try:
        with open(template_path, "r") as file:
            template = file.read()
        logging.info(f"Template {template_name} loaded successfully.")
        return template
    except FileNotFoundError:
        logging.error(f"Template file not found: {template_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading template {template_name}: {e}")
        raise

def _get_prompt_template(template_name: str, **kwargs: Any) -> HumanMessagePromptTemplate:
    """
    Creates a HumanMessagePromptTemplate based on the provided template name and parameters.

    Args:
        template_name (str): The name of the template.
        **kwargs: Additional parameters for the template.

    Returns:
        HumanMessagePromptTemplate: The configured prompt template.

    Raises:
        ValueError: If the template name is invalid.
    """
    template_configs = {
        "initialsql": {"input_variables": ["SCHEMA_STR", "QUESTION_TEXT"]},
        "intend_detection_dummysql": {"input_variables": ["QUESTION_TEXT"]},
        "schema_linking": {"input_variables": ["QUESTION_TEXT","SCHEMA_STR","LIST_WORD_TOKENS"]},
        "gen_ambiqt_sql": {"input_variables": ["QUESTION_TEXT","SCHEMA_STR","HINT"]},
        "gen_many_sql_bird": {"input_variables": ["QUESTION_TEXT","SCHEMA_STR","HINT"]},
        "gen_many_sql_spider": {"input_variables": ["QUESTION_TEXT","SCHEMA_STR","HINT"]},
        "rewrite": {"input_variables": ["SCHEMA_STR", "MATCHED_CONTENTS", "QUESTION_TEXT","EXISTING_SQL_QUERY"]},
        "select_final": {"input_variables": ["SCHEMA_STR", "QUESTION_TEXT","EXISTING_SQL_QUERY"]},
        "select_final_bird": {"input_variables": ["SCHEMA_STR", "QUESTION_TEXT","EXISTING_SQL_QUERY"]},
        "select_final_spider": {"input_variables": ["SCHEMA_STR", "QUESTION_TEXT","EXISTING_SQL_QUERY"]},
        "rewrite_spiderbird": {"input_variables": ["SCHEMA_STR", "MATCHED_CONTENTS", "QUESTION_TEXT", "EXISTING_SQL_QUERY"]},
        "rewrite_bird": {"input_variables": ["SCHEMA_STR", "MATCHED_CONTENTS", "QUESTION_TEXT", "EXISTING_SQL_QUERY"]},
        "rewrite_spider": {"input_variables": ["SCHEMA_STR", "MATCHED_CONTENTS", "QUESTION_TEXT", "EXISTING_SQL_QUERY"]},
    }

    if template_name not in template_configs:
        raise ValueError(f"Invalid template name: {template_name}")

    config = template_configs[template_name]
    input_variables = config["input_variables"]
    partial_variables = config.get("partial_variables", {})

    template_content = load_template(template_name)
    
    human_message_prompt_template = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=template_content,
            input_variables=input_variables,
            partial_variables=partial_variables
        )
    )

    return human_message_prompt_template

def get_prompt(template_name: str, schema_string: str = None) -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate based on the provided template name and schema string.

    Args:
        template_name (str): The name of the template.
        schema_string (str, optional): The schema string for the template. Defaults to None.

    Returns:
        ChatPromptTemplate: The combined prompt template.
    """
    human_message_prompt_template = _get_prompt_template(template_name=template_name, schema_string=schema_string)
    
    combined_prompt_template = ChatPromptTemplate.from_messages(
        [human_message_prompt_template]
    )
    
    return combined_prompt_template
