"""
From result gen in DIN-SQL.py,
postprocess to extract SQL queries

"""

import re
import json

def parse_response_sql(response):
    """
    Parse the response to extract SQL queries.
    :param response: The response containing SQL queries.
    :return: A list of SQL queries.
    """
    response = response.replace('```sql', '').replace('```', '')
    if "SQL: " in response:
        response = response.split("SQL: ")[1]
    sql_list = response.split(';')
    sql_list = [sql.strip() for sql in sql_list if sql.strip()]
    return sql_list
def extract_sql_queries(text):
    if "SQL: " in text:
        text = text.split("SQL: ")[1]
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
    sql_pattern = re.compile(r"(SELECT.*?[;]|SELECT.*?[```]|SELECT.*?,')", re.DOTALL | re.IGNORECASE)
    # Find all matches
    matches = sql_pattern.findall(text)
    # Clean up whitespace
    # queries = [m.strip('` \n') for m in matches if len(m.strip()) > 10] # only keep sufficiently long matches
    queries = [m.strip('` \n') for m in matches] # only keep sufficiently long matches
    return queries
if __name__ == '__main__':
    # result_file = "predicted_sql.txt"
    result_file = "predicted_table_sql.json"
    DINSQL_result = json.load(open(result_file, 'r'))
    # output_file = "predicted_sql_postprocess.json"
    output_file = "predicted_table_sql_postprocess.json"
    for i in range(len(DINSQL_result)):
        if "DINSQL_queries_text" in DINSQL_result[i]:
            sql = DINSQL_result[i]["DINSQL_queries_text"]
            # sql_list = parse_response_sql(sql)
            queries = extract_sql_queries(sql)
            # print(sql_list)
            print(len(queries),queries)
            if len(queries) == 0:
                queries = ["SELECT * FROM DummyTable WHERE DummyColumn = 'DummyValue'"]  # Fallback query if none found
            DINSQL_result[i]["queries"] = queries
            # import pdb; pdb.set_trace()
    json.dump(DINSQL_result, open(output_file, 'w'), indent=4, ensure_ascii=False)
"""
python postprocess_result.py
"""