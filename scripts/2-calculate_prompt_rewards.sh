
source scripts/.env
export CUDA_LAUNCH_BLOCKING=1
export ACCELERATE_LOG_LEVEL=info
export NCCL_DEBUG=INFO

eval "$(conda shell.bash hook)"
conda activate llama-factory

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
	comm -23 \
		<(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
)"
NNODES=$SLURM_NNODES
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: ${#GPUS[@]}"
GPUS_PER_NODE=${#GPUS[@]}
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

accelerate_config_path=scripts/accelerate_config.yaml
ds_config_path=scripts/ds_config.json

declare -a dataset_split=(
    "train"
    # "test"
)

model_name_or_path=$MISTRAL_MODEL
model_name_or_path=output/dpo/mistral_7b-adversarial-raw-sigmoid
model_name_or_path=output/dpo/mistral_7b-adversarial-raw-sigmoid-adversarial-0-defender-sigmoid
model_name_or_path=output/dpo/mistral_7b-adversarial-raw-sigmoid-adversarial-0-defender-sigmoid-adversarial-1-defender-sigmoid
model_name_or_path=output/dpo/mistral_7b-adversarial-raw-sigmoid-adversarial-0-defender-sigmoid-adversarial-1-defender-sigmoid-adversarial-2-defender-sigmoid
model_base_name=$(echo $model_name_or_path | awk -F'/' '{print $NF}')
dataset_name=iterate-3
for split in "${dataset_split[@]}"; do
    accelerate launch \
        --multi_gpu \
        --config_file ${accelerate_config_path} \
        --main_process_ip ${MASTER_ADDR} \
        --num_processes ${NUM_PROCESSES} \
        --num_machines ${NNODES} \
    scripts/calc_prompt_reward.py \
        --stage sft \
        --model_name_or_path $model_name_or_path \
        --raw_data_path data/adversarial_dataset/${dataset_name}/prompt-mistral_7b-adversarial-0-attacker-sigmoid-adversarial-1-attacker-sigmoid-adversarial-2-attacker-sigmoid/ \
        --save_name data/adversarial_dataset/${dataset_name}/score-$model_base_name/ \
        --batch_size 8 \
        --preprocessing_num_workers 8 \
        --dataloader_num_workers 8 \
        --template mistral \
        --cutoff_len 2048 \
        --bf16 \
        --flash_attn off \
        --beta 1 \

done
