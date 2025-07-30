# RAQS-SQL : Resolving Ambiguities in Text-to-SQL Systems

Code for paper: Resolving Ambiguities in Text-to-SQL Systems

We propose the RAQS-SQL framework to Resolve Ambiguities in QuestionS for Text-to-SQL system.
To handle schema-level ambiguity, we use a model that aligns the query intent directly with relevant database columns. To handle value ambiguity, we introduce techniques leveraging semantic similarities and hierarchical entity relationships of value entities stored in the database and in the question.

## Setup environment

```shell
conda create -n raqs-sql python=3.10.14
conda activate raqs-sql
pip install -r requirements.txt
```

## Prepare dataset

### Download the dataset

AmbiVal dataset can be downloaded from here [AmbiVal dataset](https://huggingface.co/datasets/griffith-bigdata/ambival). Store dataset in `misc/dataset/AmbiVal/ambival`

```shell
python preprocessing/download.py --type ambival
cd misc/dataset/ambival/
tar -xvf database_test.tar 
tar -xvf ext_info.tar 
```

Spider and bird preprocess

```shell
python preprocessing/download.py --type spiderbird
```

Spider database can be download from here [Spider dataset](https://yale-lily.github.io/spider). 
Store database in `misc/dataset/SpiderBIRD_dataset/`

```shell
cd misc/dataset/SpiderBIRD_dataset
unzip spider_data.zip -d spider
```

BIRD database can be download from here [BIRD dataset](https://bird-bench.github.io/). 
Store database in `misc/dataset/SpiderBIRD_dataset/`

```shell
cd misc/dataset/SpiderBIRD_dataset
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
unzip dev.zip -d bird
cd bird/dev_20240627/
unzip dev_databases.zip 
```

### Download pre-trained models

Download relational classification and schema linking models

```shell
python preprocessing/download.py --type model
```

### Prepare embeddings 

AmbiVal 

```shell
rm -rf misc/dataset/AmbiVal/ambival_embedding
python preprocessing/embedding.py --dataset ambival
```

Spider 

```shell
rm -rf misc/dataset/SpiderBIRD_dataset/spider_embedding
python preprocessing/embedding.py --dataset spider
```

BIRD

```shell
rm -rf misc/dataset/SpiderBIRD_dataset/bird_embedding
python preprocessing/embedding.py --dataset bird
```

## Run 

Export api-key 

```shell
export OPENAI_API_KEY=your_openai_api_key
export TOGETHER_API_KEY=your_together_api_key
export NVIDIA_API_KEY=your_nvidia_api_key
```

Run pipeline for AmbiVal dataset

```shell
python run.py --config_file=configs/api_qwen25_ambival.yaml  --output_path outputs/ambival_qwen25.jsonl
```

Spider 

```shell
python run.py --config_file=configs/api_qwen25_spider_dev.yaml  --output_path outputs/spider_qwen25.jsonl
```

Bird

```shell
python run.py --config_file=configs/api_qwen25_bird_dev.yaml  --output_path outputs/bird_qwen25.jsonl
```

## Eval

```shell
python eval/evaluate.py --dataset ambival --output_path outputs/ambival_qwen25.jsonl
```

