
# preprocess test set
echo "preprocessing..."

python src/preprocessing.py \
    --mode "test" \
    --table_path "/mnt/tampm/data_text2sql/spider_data/tables.json" \
    --input_dataset_path "/mnt/tampm/AmbiQT/benchmark/col-synonyms/validation.json" \
    --output_dataset_path "./generate_datasets/preprocessed_AmbiQT_col_data.json" \
    --db_path "/mnt/tampm/data_text2sql/spider_data/database/" \
    --target_type "sql"

python src/preprocessing.py \
    --mode "test" \
    --table_path "/mnt/tampm/data_text2sql/spider_data/tables.json" \
    --input_dataset_path "/mnt/tampm/AmbiQT/benchmark/tbl-synonyms/validation.json" \
    --output_dataset_path "./generate_datasets/preprocessed_AmbiQT_tbl_data.json" \
    --db_path "/mnt/tampm/data_text2sql/spider_data/database/" \
    --target_type "sql"
# recall tables
echo "recall tables..."

python src/table_recall.py \
    --input_dataset_path "./generate_datasets/preprocessed_AmbiQT_col_data.json" \
    --output_recalled_tables_path "./generate_datasets/table_recall_AmbiQT_col.json" --n 3 \

python src/table_recall.py \
    --input_dataset_path "./generate_datasets/preprocessed_AmbiQT_tbl_data.json" \
    --output_recalled_tables_path "./generate_datasets/table_recall_AmbiQT_tbl.json" --n 3 \
# recall columns
echo "recall columns..."

python src/column_recall.py \
    --input_recalled_tables_path "./generate_datasets/table_recall_AmbiQT_col.json" \
    --output_recalled_columns_path "./generate_datasets/column_recall_AmbiQT_col.json" --n 3 \


python src/column_recall.py \
    --input_recalled_tables_path "./generate_datasets/table_recall_AmbiQT_tbl.json" \
    --output_recalled_columns_path "./generate_datasets/column_recall_AmbiQT_tbl.json" --n 3

echo "generate prompt..."

python src/prompt_generate.py \
    --input_dataset_path "./generate_datasets/column_recall_AmbiQT_col.json" \
    --output_dataset_path "./generate_datasets/C3_dev_AmbiQT_col.json"

python src/prompt_generate.py \
    --input_dataset_path "./generate_datasets/column_recall_AmbiQT_tbl.json" \
    --output_dataset_path "./generate_datasets/C3_dev_AmbiQT_tbl.json"