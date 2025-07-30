"""
Convert a JSONL file to a JSON file.

"""

import json
import os


if __name__ == "__main__":
    # jsonl_path = "col_output_qwen_qwen2.5-coder-32b-instruct_bk.jsonl"  # Path to the input JSONL file
    jsonl_path = "col_output_meta_llama-3.3-70b-instruct_bk.jsonl"  # Path to the input JSONL file
    # json_path = "col_output_qwen_qwen2.5-coder-32b-instruct_bk.json"      # Path to the output JSON file
    json_path = "col_output_meta_llama-3.3-70b-instruct_bk.json"      # Path to the output JSON file

    if not os.path.exists(jsonl_path):
        print(f"Error: The file {jsonl_path} does not exist.")
        exit(1)
    data = []
    with open(jsonl_path, 'r') as jsonl_file:
        for line in jsonl_file:
            item = json.loads(line)
            if item['queries'] == []:
                item['queries'] = ["SELECT * FROM table;"]  # Default query if empty
            data.append(item)
                # print(f"Warning: Empty queries found in line: ")

    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Converted {jsonl_path} to {json_path}.")
"""
python cvt_jsonl2json.py
"""