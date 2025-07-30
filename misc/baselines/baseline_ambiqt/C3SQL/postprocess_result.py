"""
From result gen in DIN-SQL.py,
postprocess to extract SQL queries

"""

import os
import re
import json

from src.get_selfconsistent_output import postprocess


def convert_ambiQT_to_dict(ambiQT_data):
    """
    Convert AmbiQT data to dict
    :param ambiQT_data:
    :return: dict with key is question, value is AmbiQT item
    """
    ambiQT_question_dict = {}
    for item in ambiQT_data:
        ambiQT_question_dict[item["question"]] = item
    return ambiQT_question_dict


if __name__ == '__main__':
    # data_amb_root = "/mnt/tampm/AmbiQT/benchmark/col-synonyms/"
    data_amb_root = "/mnt/tampm/AmbiQT/benchmark/tbl-synonyms/"
    dataset_amb_path = "validation.json"
    ambiQT_data = json.load(open(os.path.join(data_amb_root,dataset_amb_path)))
    ambiQT_question_dict = convert_ambiQT_to_dict(ambiQT_data)

    # result_file = "predicted_sql_ambiQT.json"
    result_file = "predicted_sql_ambiQT_tbl.json"
    C3SQL_result = json.load(open(result_file, 'r'))
    # output_file = "predicted_col_sql_postprocess.json"
    output_file = "predicted_tbl_sql_postprocess.json"
    C3SQL_postprocessed_data = []
    # output_file = "predicted_table_sql_postprocess.json"
    for item in C3SQL_result:
        if item['question'] in ambiQT_question_dict:
            item["orig_query"] = ambiQT_question_dict[item['question']]["orig_query"]
            # item["extra_map"] = ambiQT_question_dict[item['question']]["extra_map"]
            item["extra_table_map"] = ambiQT_question_dict[item['question']]["extra_table_map"]
            C3SQL_queries_text = item["C3SQL_queries_text"]
            C3SQL_queries_text_postprocessed = []
            for queries in C3SQL_queries_text:
                queries = queries.replace('```sql', '').replace('```', '')
                if "SQL: " in queries:
                    queries = queries.split("SQL: ")[1]
                # Pattern to find SQL queries (handles code blocks and inline statements)
                sql_pattern = re.compile(r"(SELECT.*?[;]|SELECT.*?[```]|SELECT.*?,')", re.DOTALL | re.IGNORECASE)
                # Find all matches
                matches = sql_pattern.findall(queries)
                # Clean up whitespace
                queries = [m.strip('` \n') for m in matches]
                C3SQL_queries_text_postprocessed.extend(queries)
            item["C3SQL_queries_text"] = C3SQL_queries_text_postprocessed
            C3SQL_postprocessed_data.append(item)

    json.dump(C3SQL_postprocessed_data, open(output_file, 'w'), indent=4, ensure_ascii=False)
"""
python postprocess_result.py
"""

"""
python benchmark/col-synonyms/evaluate.py -f ../C3SQL/predicted_col_sql_postprocess.json  -p C3SQL_queries_text -k 5
python benchmark/tbl-synonyms/evaluate.py -f ../C3SQL/predicted_tbl_sql_postprocess.json  -p C3SQL_queries_text -k 5
"""