python finetune.py \
    --dataset_path ../data/chatglm-tuning/sens-chat-single-all \
    --lora_rank 16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --max_steps 2000 \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 20 \
    --output_dir ../model/chatglm-tuning/sens-chat-single-all-0706-2