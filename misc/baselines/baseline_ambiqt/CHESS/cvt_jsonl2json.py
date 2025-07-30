"""
Chuyển dữ liệu từ định dạng JSONL sang JSON để eval

"""

import json
import os

if __name__ == '__main__':
    # input_file = "results/dev/CHESS_IR_SS_CG_qwen/col-synonyms/ambiqt.jsonl"
    # output_file = input_file.split("/")[-1].replace(".jsonl","_qwen_col_fromjsonl.json")
    # input_file = "results/dev/CHESS_IR_SS_CG_qwen/tbl-synonyms/ambiqt.jsonl"
    # output_file = input_file.split("/")[-1].replace(".jsonl", "_qwen_tbl_fromjsonl.json")
    # input_file = "results/dev/CHESS_IR_SS_CG_llama33/col-synonyms/ambiqt.jsonl"
    # output_file = input_file.split("/")[-1].replace(".jsonl", "_llama33_col_fromjsonl.json")
    input_file = "results/dev/CHESS_IR_SS_CG_llama33/tbl-synonyms/ambiqt.jsonl"
    output_file = input_file.split("/")[-1].replace(".jsonl", "_llama33_tbl_fromjsonl.json")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found")
    print(output_file)
    # exit()
    eval_tuple_lst = []
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            item = json.loads(line.strip())
            if len(item['queries']) == 0:
                item['queries'] = ["SELECT * FROM table"]
            # import pdb; pdb.set_trace()
            eval_tuple_lst.append(item)

    with open(output_file, 'w', encoding='utf-8') as fout:
        json.dump(eval_tuple_lst, fout, ensure_ascii=False, indent=2)
        print(f"Converted JSONL file to JSON format and saved to {output_file}")
"""
python cvt_jsonl2json.py
"""
"""
python benchmark/col-synonyms/evaluate.py --file ../CHESS/ambiqt_qwen_col_fromjsonl.json -p queries -k 10
python benchmark/tbl-synonyms/evaluate.py --file ../CHESS/ambiqt_qwen_tbl_fromjsonl.json -p queries -k 10
"""