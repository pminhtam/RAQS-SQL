
# preprocess data
bash scripts/prepare_dataset.sh
# run prediction
python src/generate_sqls_by_gpt.py --input_dataset_path "./generate_datasets/C3_dev_AmbiQT_col.json"  --output_dataset_path "predicted_sql_ambiQT.json" --db_dir "/mnt/tampm/data_text2sql/spider_data/database/"

python src/generate_sqls_by_gpt.py --input_dataset_path "./generate_datasets/C3_dev_AmbiQT_tbl.json"  --output_dataset_path "predicted_sql_ambiQT_tbl.json" --db_dir "/mnt/tampm/data_text2sql/spider_data/database/"
