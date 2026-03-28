CUDA_VISIBLE_DEVICES=0 python tsv_main.py  --model_name llama3.1-8B  --dataset_name tqa --most_likely 1 --num_gene 1 --gene 1  > gen.log 2>&1 &

