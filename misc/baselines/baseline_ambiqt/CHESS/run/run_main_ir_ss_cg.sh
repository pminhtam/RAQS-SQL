source .env
data_mode=$DATA_MODE # Options: 'dev', 'train' 
data_path=$DATA_PATH # UPDATE THIS WITH THE PATH TO THE TARGET DATASET

config="./run/configs/CHESS_IR_SS_CG.yaml"

num_workers=1 # Number of workers to use for parallel processing, set to 1 for no parallel processing

#python3 -u ./src/main.py --data_mode ${data_mode} --data_path ${data_path} --config "$config" \
#        --num_workers ${num_workers} --pick_final_sql true

export DB_ROOT_PATH=/mnt/tampm/data_text2sql/spider_data/database
python3 -u ./src/main.py --data_mode 'dev' --data_path "/mnt/tampm/data_text2sql/spider_data/dev.json" \
 --config "./run/configs/CHESS_IR_SS_CG_qwen.yaml"    --num_workers 1--pick_final_sql true


## Qwen-2.5-coder-32b-instruct
python3 -u ./src/main.py --data_mode 'dev' --data_path "/mnt/tampm/AmbiQT/benchmark/col-synonyms/validation.json" \
 --config "./run/configs/CHESS_IR_SS_CG_qwen.yaml"    --num_workers 1 --pick_final_sql true

python3 -u ./src/main.py --data_mode 'dev' --data_path "/mnt/tampm/AmbiQT/benchmark/tbl-synonyms/validation.json" \
   --config "./run/configs/CHESS_IR_SS_CG_qwen.yaml"    --num_workers 1 --pick_final_sql true

## meta/llama-3.3-70b-instruct

python3 -u ./src/main.py --data_mode 'dev' --data_path "/mnt/tampm/AmbiQT/benchmark/col-synonyms/validation.json" \
 --config "./run/configs/CHESS_IR_SS_CG_llama33.yaml"    --num_workers 1 --pick_final_sql true

python3 -u ./src/main.py --data_mode 'dev' --data_path "/mnt/tampm/AmbiQT/benchmark/tbl-synonyms/validation.json" \
   --config "./run/configs/CHESS_IR_SS_CG_llama33.yaml"    --num_workers 1 --pick_final_sql true



## llama-33-together


python3 -u ./src/main.py --data_mode 'dev' --data_path "/mnt/tampm/AmbiQT/benchmark/tbl-synonyms/validation.json" \
   --config "./run/configs/CHESS_IR_SS_CG_llama33_together.yaml"    --num_workers 1 --pick_final_sql true