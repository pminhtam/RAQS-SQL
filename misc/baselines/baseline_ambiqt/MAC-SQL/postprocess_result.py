"""
From result gen in DIN-SQL.py,
postprocess to extract SQL queries

"""

import re
import json
import argparse

def extract_sql_queries(text):
    # if "SQL: " in text:
    #     text = text.split("SQL: ")[1]
    if "```sql" in text:
        text = text.split("```sql")[1]
    # Pattern to find SQL queries (handles code blocks and inline statements)
    # sql_pattern = re.compile(
    #     r"""
    #     (?:```(?:sql)?\s*)?                   # Optional code block opening
    #     (
    #         (?:(?:SELECT)   # Start with typical SQL commands
    #         [\s\S]+?                            # Non-greedy capture of everything
    #         :(?:;|',)?)                                 # Until the first ; (optional, as not all queries end with ;)
    #     )
    #     (?:\s*```)?                            # Optional code block closing
    #     """,
    #     re.IGNORECASE | re.VERBOSE
    # )
    # sql_pattern = re.compile(r"(SELECT[^\"]*;[^\"]*)", re.DOTALL | re.IGNORECASE)
    # sql_pattern = re.compile(r"(SELECT[^\"]*?[;,\']+)", re.DOTALL | re.IGNORECASE)
    # sql_pattern = re.compile(r"(SELECT.*?[;]|SELECT.*?[```]|SELECT.*?,')", re.DOTALL | re.IGNORECASE)
    sql_pattern = re.compile(r"(SELECT.*?[;]|SELECT.*?[]]|SELECT.*?[```]|SELECT.*?,')", re.DOTALL | re.IGNORECASE)
    # sql_pattern = re.compile(r"(SELECT.*?[.;]]|SELECT.*?```|SELECT.*?[;]|SELECT.*?[]]|SELECT.*?,')", re.DOTALL | re.IGNORECASE)
    # sql_pattern = re.compile(r"(SELECT.*?[;.\]])", re.DOTALL | re.IGNORECASE)
    # Find all matches
    matches = sql_pattern.findall(text)
    # queries = []
    # for m in matches:
        # Clean up whitespace and remove trailing semicolons or backticks
        # matches_sub = sql_pattern.findall(m)
        # queries.extend([m.strip('` \n').replace("\n"," ").replace("]",";").replace(".",";") for m in matches_sub]) # only keep sufficiently long matches
    # import pdb; pdb.set_trace()
    # Clean up whitespace
    # queries = [m.strip('` \n') for m in matches if len(m.strip()) > 10] # only keep sufficiently long matches
    queries = [m.strip('` \n').replace("]",";") for m in matches] # only keep sufficiently long matches
    return queries
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='path to dataset input')
    parser.add_argument('--output_file', type=str, required=True, help='path to predicted output')
    args = parser.parse_args()
    # result_file = "predicted_sql.txt"
    result_file = args.input_file
    output_file = args.output_file
    MACSQL_result = []
    with open(result_file, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                queries_1 = extract_sql_queries(item['ori_decomposer_reply'])
                queries_2 = extract_sql_queries(item['pred'])
                queries_1.extend(queries_2)
                item['queries'] = queries_1
                MACSQL_result.append(item)
                # import pdb; pdb.set_trace()
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
        # if "DINSQL_queries_text" in MACSQL_result[i]:
        #     sql = MACSQL_result[i]["DINSQL_queries_text"]
        #     queries = extract_sql_queries(sql)
        #     # print(sql_list)
        #     print(len(queries),queries)
        #     if len(queries) == 0:
        #         queries = ["SELECT * FROM DummyTable WHERE DummyColumn = 'DummyValue'"]  # Fallback query if none found
        #     MACSQL_result[i]["queries"] = queries
        #     # import pdb; pdb.set_trace()
    json.dump(MACSQL_result, open(output_file, 'w'), indent=4, ensure_ascii=False)
"""
python postprocess_result.py --input_file outputs/spider/output_spider_col_syn_qwen.json --output_file outputs/spider/postprocessed_output_spider_col_syn_qwen.json
python benchmark/col-synonyms/evaluate.py --file ../MAC-SQL/outputs/spider/postprocessed_output_spider_col_syn_qwen.json -p queries -k 10


"""