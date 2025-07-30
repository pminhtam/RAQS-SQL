"""

Step 2 : Tìm trong embedding database, các value tương đồng với value trong câu hỏi


"""
from typing import Dict
import threading
import chromadb
from chromadb.utils import embedding_functions
import re
import time
from src.utils.state import GraphState
from src.utils.tool import Tool
from sqlglot import exp, parse_one
import sqlglot
from sqlglot.optimizer.scope import build_scope
from sqlglot.optimizer.scope import find_all_in_scope
from sqlglot.optimizer.qualify import qualify


class EmbeddingSearch(Tool):
    """
    Tool for searching relevant values in the embedding database based on the SQL queries.
    """
    def __init__(self, embedding_path:str, model_embedding_name:str,str_emb:str = "{value}"):
        super().__init__()

        self.embedding_path = embedding_path            # "/mnt/tampm/EmbeddingTaskAmbiText2SQL/output_embedding_type2"
        self.model_embedding_name = model_embedding_name    # "all-MiniLM-L6-v2"
        self.str_emb = str_emb              # "{value}"

        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.model_embedding_name,
                                                                                           device="cuda:0")
        self.client = chromadb.PersistentClient(path=f"{self.embedding_path}/")

    def parse_linking_val2col(self, sql_query: str) -> dict:
        """
            Parse SQL query to get tables, columns, values
            Using sqlglot library

            :param sql_query:
            :return:
            """
        try:
            sql_query = sql_query.replace("\"", "'")
            """
            Replace " by ' to avoid error when parsing.
            sqlglot nhầm giữa tên column và giá trị string
            """
            """
            BUG22122024 : lỗi khi SQL có kí tự "`" trong câu SQL. SQLGLot parse không đúng
            """
            if '`' in sql_query:
                ast = sqlglot.parse_one(sql_query, read="mysql")
            else:
                ast = sqlglot.parse_one(sql_query)
        except Exception as e:
            return {"table": [], "column": [], "comlumn_dict": {}, "value_list": [], "value_dict": {},
                    "dependencies_dict": {}, "table_alias_dict": {}}
        try:
            qualify(ast)
        except Exception as e:
            # print(e)
            return {"table": [], "column": [], "comlumn_dict": {}, "value_list": [], "value_dict": {},
                    "dependencies_dict": {}, "table_alias_dict": {}}
        root = build_scope(ast)
        table_list = []
        table_alias_dict = {}
        value_list = []
        value_dict = {}
        column_list = []
        comlumn_dict = {}
        dependencies_dict = {}

        for column in find_all_in_scope(root.expression, exp.Column):  # Lấy dictionary : {table_name: [column_name]}
            # print(f"{column} => {root.sources[column.table]}")
            try:
                column_list.append(root.sources[column.table].name + "." + column.name)  # Lấy danh sách column
                if root.sources[column.table].name in comlumn_dict:
                    comlumn_dict[root.sources[column.table].name].append(column.name)
                else:
                    comlumn_dict[root.sources[column.table].name] = [column.name]  # Lấy danh sách column
            except Exception as e:
                # print(e)
                column_list.append(column.table + "." + column.name)
        for table in ast.find_all(exp.Table):  # Lấy danh sách table
            table_list.append(table.name)
            table_alias_dict[table.alias] = table.name
        for value in ast.find_all(exp.Literal):  # Lấy danh sách value
            value_list.append(value.name)
            value_parent = value.parent.this
            if value_parent:
                if hasattr(value_parent, "table"):
                    if value_parent.table in table_alias_dict:
                        value_dict[value.name] = {"table": table_alias_dict[value_parent.table],
                                                  "column": value_parent.name}
                    else:
                        value_dict[value.name] = {"table": value_parent.table, "column": value_parent.name}
                    """
                    Trường hợp value_parent.table == "" méo hiểu chạy kiểu gì 
                    """
                    # if value_parent.table == "":
                    #     import pdb; pdb.set_trace()

        for cte in ast.find_all(exp.CTE):
            dependencies_dict[cte.alias_or_name] = []

            cte_query = cte.this.sql()
            for table in parse_one(cte_query).find_all(exp.Table):
                dependencies_dict[cte.alias_or_name].append(table.name)
        # import pdb; pdb.set_trace()
        return {"table": table_list, "column": column_list, "comlumn_dict": comlumn_dict,
                "value_list": value_list, "value_dict": value_dict, "dependencies_dict": dependencies_dict,
                "table_alias_dict": table_alias_dict}

    def _run(self, state: GraphState):
        print("embedding_search run EmbeddingSearch linking")
        sql_queries = []
        start_time = time.time()
        for previous_step in state["pipeline_log"]:
            # print(f"previous_step : {previous_step['step']}")
            # if previous_step["step"] == "initial_sql":
            #     sql_queries = previous_step["response"]['SQL']
            #     break
            # elif previous_step["step"] == "gen_ambiqt_sql":
            #     queries_list = previous_step["queries_list"]['queries']
            #     sql_queries = queries_list[0]
            #     print(f"sql_queries : {sql_queries}")
            #     # sql_queries = queries_list
            #     break
            if "sql_preds" in previous_step:
                sql_queries.extend([sql_pred["sql"] for sql_pred in previous_step["sql_preds"]])
                # print(f"sql_queries : {sql_queries}")

        if isinstance(sql_queries, list):
            """
            Nếu có nhiều câu SQL thì parse từng câu SQL
            Sau đó gộp các giá trị lại với nhau
            """
            sql_result_parsed = {}
            for idx, sql_query in enumerate(sql_queries):
                sql_result_parsed_item = self.parse_linking_val2col(sql_query)
                for val in sql_result_parsed_item:
                    if val not in sql_result_parsed:
                        sql_result_parsed[val] = sql_result_parsed_item[val]

        elif isinstance(sql_queries, str):
            sql_result_parsed = self.parse_linking_val2col(sql_queries)
        else:
            sql_result_parsed = {"value_dict": {}}
            # raise ValueError("sql_queries must be a string or a list of strings")
        db_id = state["item"].db_id
        try:
            val_linking_emb = self.embedding_search(db_id, sql_result_parsed)
        except Exception as e:
            print(f"Error in embedding_search: {e}")
            val_linking_emb = {}
            # import pdb; pdb.set_trace()
        # if len(val_linking_emb) == 0:
        # import pdb; pdb.set_trace()
        end_time = time.time()
        run_time = end_time - start_time
        state["pipeline_log"].append({
            "step": self.tool_name,
            "name": "embedding_search",
            "val_linking_emb": val_linking_emb,
            "sql_result_parsed": sql_result_parsed,
            "run_time": run_time

        })


    def embedding_search(self,db_id, sql_result_parsed):
        # embedding_path = "output_embedding_update"
        """
        File embedding là output của script embedding/embedding_ambval.py
        """
        # client = chromadb.PersistentClient(path=f"{embedding_path}/{db_id}/{table}")
        val_linking_emb = {}
        pattern = re.compile(r'\d+-\d+-\d+')        # wikisql match pattern
        db_id_ori = db_id
        for val in sql_result_parsed["value_dict"]:
            table_que_ori = sql_result_parsed["value_dict"][val]["table"].lower()
            if table_que_ori == db_id_ori and pattern.match(db_id_ori):
                table_que = "wikisql"
                db_id = "wikisql"
            else:
                table_que = table_que_ori
                db_id = db_id_ori
            # if table != table_que:
            #     continue
            column_que = sql_result_parsed["value_dict"][val]["column"]
            column_lower = column_que.lower()
            column_lower = column_lower.replace(" ", "z").replace("/", "z").replace("(", "z").replace(")", "z").replace(
                "%", "z").replace("'", "z").replace("=", "z").replace(",", "z") + "z"  # Tên đổi từ script embedding_ambval.py
            collection_name = (db_id+table_que)[:(63-len(column_lower))]+column_lower
            try:
                collection = self.client.get_collection(collection_name, embedding_function=self.sentence_transformer_ef)
            except:
                print(f"Collection {collection_name} not found")
                continue
            relevant_docs = collection.query(query_texts=self.str_emb.format(table=table_que, column=column_que, value=val), n_results=10)
            metadata = relevant_docs['metadatas'][0]
            val_linking_emb[val] = {
                "candidate": [],
                "table": table_que,
                "column": column_que,
            }
            for idx, doc in enumerate(metadata):
                val_linking_emb[val]['candidate'].append(doc["value"])
                # print(f"{idx} : {doc['value']}")

        return val_linking_emb

