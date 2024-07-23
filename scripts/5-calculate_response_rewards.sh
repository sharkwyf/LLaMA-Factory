
source scripts/.env
export CUDA_LAUNCH_BLOCKING=1
export ACCELERATE_LOG_LEVEL=info

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
    "test"
)

model_name_or_path=/mnt/petrelfs/wangyuanfu/workspace/AdversarialLLM/output/rm/mistral_7b-adversarial-raw-1
dataset_name=iterate-0
model_base_name=$(echo $model_name_or_path | awk -F'/' '{print $NF}')
for split in "${dataset_split[@]}"; do
    accelerate launch \
        --multi_gpu \
        --config_file ${accelerate_config_path} \
        --main_process_ip ${MASTER_ADDR} \
        --num_processes ${NUM_PROCESSES} \
        --num_machines ${NNODES} \
    scripts/calc_response_reward.py \
        --stage sft \
        --model_name_or_path $model_name_or_path \
        --raw_data_path data/adversarial_dataset/$dataset_name/eval-response-mistral_7b-adversarial-raw-sigmoid-adversarial-raw-sigmoid-adversarial-raw-sigmoid-adversarial-raw-sigmoid-adversarial-raw-sigmoid/ \
        --save_name data/adversarial_dataset/$dataset_name/eval-response-mistral_7b-adversarial-raw-sigmoid-adversarial-raw-sigmoid-adversarial-raw-sigmoid-adversarial-raw-sigmoid-adversarial-raw-sigmoid/ \
        --split $split \
        --batch_size 8 \
        --preprocessing_num_workers 8 \
        --dataloader_num_workers 8 \
        --template mistral \
        --cutoff_len 2048 \
        --bf16 \
        --flash_attn off \

done
