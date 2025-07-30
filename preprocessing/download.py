import os
import argparse
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        choices=["ambival", "spiderbird", "model"],
        required=True,
    )
    args = parser.parse_args()
    download_type = args.type
    if download_type == "ambival":
        snapshot_download(
            local_dir="misc/dataset/ambival/",
            repo_id="griffith-bigdata/ambival",
            repo_type="dataset",
        )
    elif download_type == "spiderbird":
        snapshot_download(
            local_dir="misc/dataset/SpiderBIRD_dataset/",
            repo_id="griffith-bigdata/SpiderBIRD_dataset",
            repo_type="dataset",
        )
    elif download_type == "model":
        snapshot_download(
            local_dir="misc/model_weights/relational_classification",
            repo_id="griffith-bigdata/relational-classification",
            repo_type="model",
        )

        snapshot_download(
            local_dir="misc/model_weights/schema_linking",
            repo_id="griffith-bigdata/schema-linking",
            repo_type="model",
        )
