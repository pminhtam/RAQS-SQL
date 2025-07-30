

"""
Source https://github.com/ShayanTalaei/CHESS/blob/main/src/llm/parsers.py
Bóc tách dữ liệu từ output của model llm sau khi gọi api

Step 3 trong quá trình gọi api


"""


import json
import re
import logging
from typing import Any, Dict, List, Tuple

from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.outputs import ChatGeneration, Generation

class BaseOutputParserWithMetadata(BaseOutputParser):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        parse_text = self.parse(result[0].text)
        response_ori = result[0].text
        num_output_tokens = result[0].message.response_metadata['token_usage']['completion_tokens']
        num_input_tokens = result[0].message.response_metadata['token_usage']['prompt_tokens']
        return parse_text,response_ori, num_input_tokens, num_output_tokens

class PythonListOutputParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing Python lists."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Any:
        """
        Parses the output to extract Python list content from markdown.

        Args:
            output (str): The output string containing Python list.

        Returns:
            Any: The parsed Python list.
        """
        logging.debug(f"Parsing output with PythonListOutputParser: {output}")
        if "```python" in output:
            output = output.split("```python")[1].split("```")[0]
        output = re.sub(r"^\s+", "", output)
        return eval(output)  # Note: Using eval is potentially unsafe, consider using ast.literal_eval if possible.
class MarkDownOutputParser(BaseOutputParserWithMetadata):
    """Parses output embedded in markdown code blocks containing SQL queries."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with MarkDownOutputParser: {output}")
        if "</think>" in output:
            output = output.split("</think>")[-1]
        if "```sql" in output:
            output = output.split("```sql")[1].split("```")[0]
        elif "```" in output:
            output = output.split("```")[1].split("```")[0]
        output = re.sub(r"^\s+", "", output)
        return {"SQL": output}

class NoneParser(BaseOutputParserWithMetadata):
    """Parses output embedded in markdown code blocks containing SQL queries."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str):
        return output
def get_parser(parser_name: str) -> BaseOutputParser:
    """
    Returns the appropriate parser based on the provided parser name.

    Args:
        parser_name (str): The name of the parser to retrieve.

    Returns:
        BaseOutputParser: The appropriate parser instance.

    Raises:
        ValueError: If the parser name is invalid.
    """
    parser_configs = {
        "column_linking": MarkDownOutputParser,
        "val_linking": JsonOutputParser,
        "rewrite": MarkDownOutputParser,
        "none" : NoneParser,
        "markdown": MarkDownOutputParser,

    }

    if parser_name not in parser_configs:
        logging.error(f"Invalid parser name: {parser_name}")
        raise ValueError(f"Invalid parser name: {parser_name}")

    logging.info(f"Retrieving parser for: {parser_name}")
    parser = parser_configs[parser_name]() if callable(parser_configs[parser_name]) else parser_configs[parser_name]
    return parser
