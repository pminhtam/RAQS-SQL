

"""
Embedding value in database
Save embedding in disk

"""

import os
import re
import json
import argparse
import chromadb
from openai import OpenAI
from tqdm import tqdm
from chromadb.utils import embedding_functions
from chromadb import Documents, EmbeddingFunction, Embeddings


class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name, api_key, api_base):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.client = OpenAI(
              api_key=self.api_key,
              base_url=self.api_base
            )
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        response = self.client.embeddings.create(input=input, model=self.model_name,
                                                   encoding_format="float",
                                                 extra_body={"input_type": "passage", "truncate": "NONE"}
                                                 # extra_body={"input_type": "query", "truncate": "NONE"}
                                                 )
        num_senten = len(input)
        embeddings = [response.data[i].embedding for i in range(num_senten)]
        return embeddings
def embedding_ambval(distinct_values_data_dict_path, chromadb_path, str_emb, model_name="all-MiniLM-L6-v2",external_knowledge_folder_path=""):

    if model_name == "text-embedding-ada-002":
        api_key = os.environ["OPENAI_API_KEY"]
        sentence_transformer_ef = embedding_functions.OpenAIEmbeddingFunction(model_name=model_name,
                                                                                           api_key=api_key)
    elif model_name == "nvidia/embed-qa-4" or model_name == "nvidia/nv-embedcode-7b-v1" or model_name == "baai/bge-m3":
        api_key = os.environ["NVIDIA_API_KEY"]
        sentence_transformer_ef = MyEmbeddingFunction(model_name=model_name,api_key=api_key,
                                                      api_base="https://integrate.api.nvidia.com/v1")
    else:
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name,
                                                                                           device="cuda:0")  # Default function
    with open(distinct_values_data_dict_path) as inf:
        distinct_values_data_dict = json.load(inf)

    if not os.path.exists(f"{chromadb_path}/"):
        os.makedirs(f"{chromadb_path}/", exist_ok=True)
    client = chromadb.PersistentClient(path=f"{chromadb_path}/")
    for db_id in distinct_values_data_dict:
        for table_name in distinct_values_data_dict[db_id]:
            for column in distinct_values_data_dict[db_id][table_name]:
                ################### For SPIDERBIRD dataset ###################
                if any(keyword in column.lower() for keyword in
                       # if any(keyword == column.lower() for keyword in
                       ["_id", " id", "url", "email", "web", "time", "phone", "date", "address",
                        "name", "number", "code", "zip"]) or column.endswith("Id")  or column.endswith("ID"):
                    print(f"Skip column {column} in table {table_name} in database {db_id} due to keyword filter")
                    continue
                #############################################################
                values_list = distinct_values_data_dict[db_id][table_name][column]
                # print(column)
                ext_know_dataset = None
                column_save_path = column.replace(" ", "_").replace("/", "_")
                external_knowledge_value_file_path = os.path.join(external_knowledge_folder_path, f'{db_id}_{table_name}_{column_save_path}.json')
                if external_knowledge_value_file_path != "" and os.path.exists(external_knowledge_value_file_path):
                    ext_know_dataset = json.load(open(external_knowledge_value_file_path,'r'))
                column_lower = column.lower()
                column_lower = column_lower.replace(" ", "z").replace("/", "z").replace("(", "z").replace(")", "z").replace("%", "z").replace("'", "z").replace("=", "z").replace(",", "z") + "z"  # chromdb không tạo được collection chứa `space`
                try:
                    collection = client.get_or_create_collection((db_id+table_name)[:(63-len(column_lower))]+column_lower, embedding_function=sentence_transformer_ef)
                except:
                    try:
                        collection = client.create_collection((db_id+table_name)[:(63-len(column_lower))]+column_lower, embedding_function=sentence_transformer_ef)
                    except Exception as e:
                        print(e)
                        # import pdb; pdb.set_trace()
                    #     continue
                print((db_id+table_name)[:(63-len(column_lower))] + column_lower)

                collection_values = []
                collection_metadatas = []
                collection_ids = []
                for idx_item, value in enumerate(values_list):
                    try:
                        ext_know_value_dict = ext_know_dataset[db_id][table_name][column][value]
                        ext_know_value = list(ext_know_value_dict.values())[0]
                    except:
                        ext_know_value = ""
                    # collection_values.append(str_emb.format(table=table_name, column=column, value=value))
                    # collection_values.append(str_emb.format(table=table_name, column=column, value=value) + " " + str(ext_know_value))
                    collection_values.append(str_emb.format(table=table_name, column=column, value=value + " " + str(ext_know_value)))
                    collection_metadatas.append({
                        "value": str(value),
                        "table": table_name,
                        "column": column,
                        "db_id": db_id
                    })
                    # import pdb; pdb.set_trace()
                    collection_ids.append(f"{idx_item}_{table_name}_{column}_{value}")
                    # if (len(collection_values) + 1) % 5000 == 0:
                    if (len(collection_values) + 1) % 512 == 0:
                        """
                        upsert :  updates existing items, or adds them if they don't yet exist.
                        """
                        collection.upsert(
                            documents=collection_values,
                            metadatas=collection_metadatas,
                            ids=collection_ids,  # unique for each doc
                        )
                        collection_values = []
                        collection_metadatas = []
                        collection_ids = []
                if len(collection_values) > 0:
                    try:
                        collection.upsert(
                            documents=collection_values,
                            metadatas=collection_metadatas,
                            ids=collection_ids,  # unique for each doc
                        )
                    except Exception as e:
                        print(e)
                        # print(db_id, table_name, column) # world_1 city district
                        # import pdb; pdb.set_trace()
                print(db_id, table_name, column)
    del sentence_transformer_ef, client

    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True, type=str,
                        choices=["ambival", "spider","bird"])
    return parser.parse_args()


if __name__ == '__main__':

    emb_method = "{value}"
    args = parse_args()
    model_name = "all-MiniLM-L6-v2"
    dataset_name = args.dataset
    if dataset_name == "ambival":
        root_path = "misc/dataset/ambival/"
        external_knowledge_folder_path = os.path.join(root_path, "ext_info/")
    elif dataset_name == "spider" or dataset_name == "bird":
        root_path = "misc/dataset/SpiderBIRD_dataset/"
        external_knowledge_folder_path = ""
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported")
    distinct_values_data_dict_path = os.path.join(root_path,f"{dataset_name}_distinct_values.json")
    chromadb_path = os.path.join(root_path,f"{dataset_name}_embedding")

    embedding_ambval(distinct_values_data_dict_path,chromadb_path, emb_method, model_name=model_name, external_knowledge_folder_path=external_knowledge_folder_path)

