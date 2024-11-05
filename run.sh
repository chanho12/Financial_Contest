CURRENT_TIME=$(TZ=Asia/Seoul date +"%Y-%m-%d-%H.%M.%S")

MODEL_PATH='Qwen/Qwen2-7B-Instruct'
OUTPUT='/home/chanho/Model/Financial_Contest/output'

TRN_FN='/home/chanho/Model/Financial_Contest/dataset/train_data_except_own_file.csv'
DEV_FN='/home/chanho/Model/Financial_Contest/dataset/valid.csv'
mkdir -p $OUTPUT/$CURRENT_TIME

TOTAL_SIZE=$(wc -l < "${TRN_FN}")
echo "number of samples in trainset: ${TOTAL_SIZE}"
export TOKENIZERS_PARALLELISM=false

deepspeed --include localhost:2 --master_port 29504 /home/chanho/Model/Financial_Contest/main.py \
   --model_name_or_path ${MODEL_PATH} \
   --train_data_path ${TRN_FN} \
   --valid_data_path ${DEV_FN} \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --data_output_path $OUTPUT/data \
   --max_seq_len 1536 \
   --learning_rate 1e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 2 \
   --num_train_samples ${TOTAL_SIZE} \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 1000 \
   --seed 42 \
   --save_interval 2000000 \
   --eval_interval 10000 \
   --output_dir $OUTPUT/$CURRENT_TIME \
   --offload \
> $OUTPUT/$CURRENT_TIME/train.log 2>&1 &
