OUTPUT_DIR="/home/ashaw8/compute/finetunes/$RUN_NAME"

python3 create_deepspeed_config.py

deepspeed train.py \
    --model_name_or_path /home/ashaw8/compute/models/$MODEL_NAME \
    --dataset_path datasets/$TOPIC/$IDEOLOGY/$MODEL_NAME \
    --run_name $RUN_NAME \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed deepspeed_config.json \
    --max_grad_norm 1.0 \
    --tf32 False \
    --report_to wandb