#!/bin/bash
#SBATCH --job-name=llama-factory-dpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=11G # Important to enable "mix" use of GPUs across cluster users
#SBATCH --partition=llm-safety
#SBATCH --gres=gpu:4
#SBATCH --output=output/dpo/logs/%x-%j.out
#SBATCH --err=output/dpo/logs/%x-%j.err

source scripts/.env

export CUDA_HOME=/mnt/cache/share/cuda-12.0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NET=IB

eval "$(conda shell.bash hook)"
conda activate llama-factory

STAGE=dpo
model_name_or_path=/mnt/hwfile/llm-safety/models/Mistral-7B-Instruct-v0.2
dataset_name=iterate-0-attacker
# dataset_name=hh_rlhf_harmless
pref_loss=simpo
pref_beta=2.5
simpo_gamma=0.3
model_base_name=$(echo $model_name_or_path | awk -F'/' '{print $NF}')
run_name="$model_base_name-$dataset_name-$pref_loss"

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
LOG_PATH="output/$STAGE/logs/latest_run.log"

TOTAL_BATCHSIZE=512
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=$(($TOTAL_BATCHSIZE / $NNODES / $GPUS_PER_NODE / $PER_DEVICE_TRAIN_BATCH_SIZE))
# check if TOTAL_BATCHSIZE is divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE)
if [ $(($GRADIENT_ACCUMULATION_STEPS * $NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE)) -ne $TOTAL_BATCHSIZE ]; then
    echo "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE} is not divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE): $(($NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE))"
    exit 1
fi
echo "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE}, PER_DEVICE_TRAIN_BATCH_SIZE, ${PER_DEVICE_TRAIN_BATCH_SIZE}, GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}"

accelerate_config_path=scripts/accelerate_config.yaml
ds_config_path=scripts/ds_config.json
adapter_path=output/adapter/$STAGE
output_path=output/$STAGE/$run_name/

export LAUNCHER="\
accelerate launch \
    --config_file ${accelerate_config_path} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --num_processes ${NUM_PROCESSES} \
    --num_machines ${NNODES} \
    --machine_rank \$SLURM_PROCID \
"

export PROGRAM="\
src/train.py \
    --stage $STAGE \
    --do_train \
    --do_eval \
    --model_name_or_path ${model_name_or_path} \
    --dataset ${dataset_name} \
    --split train \
    --val_size 4000 \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8 \
    --overwrite_cache \
    --evaluation_strategy steps \
    --eval_steps 30 \
    --template mistral \
    --finetuning_type full \
    --output_dir ${output_path} \
    --overwrite_output_dir \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1.0 \
    --top_k 0 \
    --top_p 0.9 \
    --logging_first_step \
    --logging_steps 1 \
    --run_name $STAGE-$run_name \
    --save_steps 1000 \
    --plot_loss \
    --bf16 \
    --seed 42 \
    --flash_attn fa2 \
    --pref_loss $pref_loss \
    --pref_beta $pref_beta \
    --simpo_gamma $simpo_gamma \
    
"

export CMD="$LAUNCHER $PROGRAM"

echo "START TIME: $(date)"
echo "NNODES: ${NNODES}, GPUS_PER_NODE: ${GPUS_PER_NODE}, NUM_PROCESSES: ${NUM_PROCESSES}"
echo "MASTER_ADDR: ${MASTER_ADDR}, MASTER_PORT: ${MASTER_PORT}"
echo "PRINTENV:"
printenv

echo "CMD: ${CMD}"

if [ ! -z "$SLURM_SRUN_COMM_PORT" ]; then
    echo "This script was submitted via srun."
    bash -c "$CMD"
else
    echo "This script was submitted via sbatch."
    srun --mpi=pmi2 --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH
fi

echo "END TIME: $(date)"