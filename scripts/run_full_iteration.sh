
source scripts/.env
eval "$(conda shell.bash hook)"
conda activate llama-factory-bk

export CUDA_LAUNCH_BLOCKING=1
export ACCELERATE_LOG_LEVEL=info
# export TORCH_NCCL_ENABLE_MONITORING=0
export VLLM_ENGINE_ITERATION_TIMEOUT_S=180

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
printf "%b" "Number of GPUs: ${#GPUS[@]}"
GPUS_PER_NODE=${#GPUS[@]}
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

ACCELERATE_CONFIG_PATH=scripts/accelerate_config.yaml
DS_CONFIG_PATH=scripts/ds_config.json

rm -rf /mnt/lustre/wangyuanfu/.cache/huggingface/datasets/adversarial_dataset

seed_dataset=relabeled_hh_rlhf_harmless
harmless_prompt_ver=harmless-v1
harmful_prompt_ver=harmful-v1
if [ $seed_dataset == "hh_rlhf_harmless" ]; then
    seed_dataset_path=data/custom_dataset/hh_rlhf.py
    seed_dataset_suffix=harmless
elif [ $seed_dataset == "relabeled_hh_rlhf_harmless" ]; then
    seed_dataset_path=data/relabeled_dataset/relabeled_dataset.py
    seed_dataset_suffix=hh_rlhf_harmless
elif [ $seed_dataset == "ultra_feedback" ]; then
    seed_dataset_path=data/custom_dataset/ultra_feedback.py
    seed_dataset_suffix=all
fi

loss_type=dpo
if [ $loss_type == "dpo" ]; then
    pref_loss=sigmoid
    pref_beta=0.1  # try larger beta
    simpo_gamma=0.
    learning_rate=5.0e-7
elif [ $loss_type == "simpo" ]; then
    pref_loss=simpo
    pref_beta=2.5
    simpo_gamma=0.3
    learning_rate=5.0e-7
elif [ $loss_type == "orpo" ]; then
    pref_loss=orpo
    pref_beta=0.2
    pref_beta=1.0
    simpo_gamma=0.
    learning_rate=5.0e-7
fi

if [ ! -z "$1" ]; then
    start_iter=$1
    max_iters=$1
    start_step=1
else
    start_iter=0        # start from 0
    max_iters=30
    start_step=1
fi
total_iters=100
defender_dataset_suffix=defender
train_samples_per_iter=4096
test_samples_per_iter=300
data_path=$ADV_DATA_PATH
model_path=$ADV_MODEL_PATH
RETRY_LIMIT=3
RETRY_DELAY=60  # seconds

info="""
# Experiment Settings:
Data Path: \`$data_path\`
Model Path: \`$model_path\`
Seed dataset: \`$seed_dataset\`
Total iters: \`$total_iters\`
Training samples per iter: \`$train_samples_per_iter\`
Test samples per iter: \`$test_samples_per_iter\`
Loss type: \`$loss_type\`
Pref loss: \`$pref_loss\`
Pref beta: \`$pref_beta\`
Simpo gamma: \`$simpo_gamma\`
Learning rate: \`$learning_rate\`
Defender training data: \`$defender_dataset_suffix\`
"""
printf "%b" "$info"

mkdir -p $data_path
mkdir -p $model_path
printf "%b" "$info" > $data_path/main.md
raw_dataset_name=adversarial-raw
# reverse_raw_dataset_name=adversarial-raw-reversed
reverse_raw_dataset_name=adversarial-harmless-reversed

for iter in $(seq $start_iter $max_iters); do
    if [ $iter -eq -1 ]; then
        continue
    elif [ $iter -eq 0 ]; then
        # iter 0
        mkdir -p $data_path/base
        log_file="$data_path/base/main.log"
        printf "%b" "Running iteration $iter" | tee $log_file
        step=1

        # Step 0.1. Generate seed dataset
        attempt=0
        until [ $attempt -ge $RETRY_LIMIT ]
        do
            if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
                printf "%b" "Skipping step $step on iter $iter"
                step=$((step+1))
                break
            fi
            CMD="
            python scripts/generate_seed_prompts.py \
                --dataset_lists $seed_dataset_path $seed_dataset_suffix \
                --train_samples_per_iter $train_samples_per_iter \
                --test_samples_per_iter $test_samples_per_iter \
                --max_iters $total_iters \
                --output_dir $data_path/base/ \
            "
            echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
            printf "%b" "run_full_iteration.sh Iteration $iter, running command: Step $step: (attempt: $attempt): \"$CMD\"" | tee -a $log_file
            bash -c "$CMD"  2>&1 | tee -a $log_file

            cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
            if [ $cmd_status -eq 0 ]; then
                printf "%b" "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
                step=$((step+1))
                break
            else
                printf "%b" "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
                attempt=$((attempt+1))
                if [ $attempt -lt $RETRY_LIMIT ]; then
                    printf "%b" "Sleeping for $RETRY_DELAY seconds before retrying..." | tee -a $log_file
                    sleep $RETRY_DELAY
                else
                    printf "%b" "Reached maximum number of attempts. Exiting..." | tee -a $log_file
                    exit 1
                fi
            fi
        done

        # Step 0.2. Generate harmless prompts on adversarial-raw
        attempt=0
        until [ $attempt -ge $RETRY_LIMIT ]
        do
            if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
                printf "%b" "Skipping step $step on iter $iter"
                step=$((step+1))
                break
            fi
            CMD="
            python scripts/generate_prompts.py \
                --dataset_lists data/adversarial_dataset/adversarial_dataset.py adversarial-raw \
                --model $MISTRAL_MODEL \
                --tensor-parallel-size $GPUS_PER_NODE \
                --num_train_samples -1 \
                --num_test_samples -1 \
                --num_generated_prompts 1 \
                --prompt_ver $harmless_prompt_ver \
                --output_dir $data_path/base/harmless \
                --disable-log-requests \
            "
            echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
            printf "%b" "run_full_iteration.sh Iteration $iter, running command: Step $step: (attempt: $attempt): \"$CMD\"" | tee -a $log_file
            bash -c "$CMD"  2>&1 | tee -a $log_file

            cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
            if [ $cmd_status -eq 0 ]; then
                printf "%b" "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
                step=$((step+1))
                break
            else
                printf "%b" "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
                attempt=$((attempt+1))
                if [ $attempt -lt $RETRY_LIMIT ]; then
                    printf "%b" "Sleeping for $RETRY_DELAY seconds before retrying..." | tee -a $log_file
                    sleep $RETRY_DELAY
                else
                    printf "%b" "Reached maximum number of attempts. Exiting..." | tee -a $log_file
                    exit 1
                fi
            fi
        done

        # Step 0.3.1. Train the base attacker model on $reverse_raw_dataset_name with DPO
        attempt=999
        until [ $attempt -ge $RETRY_LIMIT ]
        do
            if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
                printf "%b" "Skipping step $step on iter $iter"
                step=$((step+1))
                break
            fi
            STAGE=dpo
            model_name_or_path=$MISTRAL_MODEL
            dataset_name=$reverse_raw_dataset_name
            attacker_run_name="auto-mistral_7b-$dataset_name-$loss_type"

            TOTAL_BATCHSIZE=512
            PER_DEVICE_TRAIN_BATCH_SIZE=4
            GRADIENT_ACCUMULATION_STEPS=$(($TOTAL_BATCHSIZE / $NNODES / $GPUS_PER_NODE / $PER_DEVICE_TRAIN_BATCH_SIZE))
            # check if TOTAL_BATCHSIZE is divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE)
            if [ $(($GRADIENT_ACCUMULATION_STEPS * $NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE)) -ne $TOTAL_BATCHSIZE ]; then
                printf "%b" "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE} is not divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE): $(($NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE))" | tee -a $log_file
                exit 1
            fi
            printf "%b" "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE}, PER_DEVICE_TRAIN_BATCH_SIZE, ${PER_DEVICE_TRAIN_BATCH_SIZE}, GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}" | tee -a $log_file

            CMD="
            accelerate launch \
                --config_file ${ACCELERATE_CONFIG_PATH} \
                --main_process_ip ${MASTER_ADDR} \
                --main_process_port ${MASTER_PORT} \
                --num_processes ${NUM_PROCESSES} \
                --num_machines ${NNODES} \
                --machine_rank \$SLURM_PROCID \
            src/train.py \
                --stage $STAGE \
                --do_train \
                --do_eval \
                --model_name_or_path ${model_name_or_path} \
                --dataset ${dataset_name} \
                --split train \
                --val_size 0.05 \
                --cutoff_len 2048 \
                --preprocessing_num_workers 4 \
                --dataloader_num_workers 8 \
                --overwrite_cache \
                --evaluation_strategy steps \
                --eval_steps 100 \
                --template mistral \
                --finetuning_type full \
                --output_dir $model_path/$attacker_run_name/ \
                --overwrite_output_dir \
                --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
                --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
                --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
                --lr_scheduler_type cosine \
                --warmup_ratio 0.1 \
                --learning_rate $learning_rate \
                --num_train_epochs 1.0 \
                --top_k 0 \
                --top_p 0.9 \
                --logging_first_step \
                --logging_steps 5 \
                --run_name $STAGE-$attacker_run_name \
                --save_steps 1000 \
                --plot_loss \
                --bf16 \
                --seed 42 \
                --flash_attn fa2 \
                --pref_loss $pref_loss \
                --pref_beta $pref_beta \
                --simpo_gamma $simpo_gamma \
            "
            echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
            printf "%b" "run_full_iteration.sh Iteration $iter, running command: Step $step: (attempt: $attempt): \"$CMD\"" | tee -a $log_file
            bash -c "$CMD"  2>&1 | tee -a $log_file
            
            cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
            if [ $cmd_status -eq 0 ]; then
                printf "%b" "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
                step=$((step+1))
                break
            else
                printf "%b" "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
                attempt=$((attempt+1))
                if [ $attempt -lt $RETRY_LIMIT ]; then
                    printf "%b" "Sleeping for $RETRY_DELAY seconds before retrying..." | tee -a $log_file
                    sleep $RETRY_DELAY
                else
                    printf "%b" "Reached maximum number of attempts. Exiting..." | tee -a $log_file
                    exit 1
                fi
            fi
        done

        # Step 0.3.2. Train the base attacker model on $reverse_raw_dataset_name with SFT
        attempt=0
        until [ $attempt -ge $RETRY_LIMIT ]
        do
            if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
                printf "%b" "Skipping step $step on iter $iter"
                step=$((step+1))
                break
            fi
            STAGE=sft
            model_name_or_path=$MISTRAL_MODEL
            dataset_name=$reverse_raw_dataset_name
            attacker_run_name="auto-mistral_7b-$dataset_name-$loss_type"

            TOTAL_BATCHSIZE=512
            PER_DEVICE_TRAIN_BATCH_SIZE=4
            GRADIENT_ACCUMULATION_STEPS=$(($TOTAL_BATCHSIZE / $NNODES / $GPUS_PER_NODE / $PER_DEVICE_TRAIN_BATCH_SIZE))
            # check if TOTAL_BATCHSIZE is divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE)
            if [ $(($GRADIENT_ACCUMULATION_STEPS * $NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE)) -ne $TOTAL_BATCHSIZE ]; then
                printf "%b" "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE} is not divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE): $(($NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE))" | tee -a $log_file
                exit 1
            fi
            printf "%b" "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE}, PER_DEVICE_TRAIN_BATCH_SIZE, ${PER_DEVICE_TRAIN_BATCH_SIZE}, GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}" | tee -a $log_file

            CMD="
            accelerate launch \
                --config_file ${ACCELERATE_CONFIG_PATH} \
                --main_process_ip ${MASTER_ADDR} \
                --main_process_port ${MASTER_PORT} \
                --num_processes ${NUM_PROCESSES} \
                --num_machines ${NNODES} \
                --machine_rank \$SLURM_PROCID \
            src/train.py \
                --stage $STAGE \
                --do_train \
                --do_eval \
                --model_name_or_path ${model_name_or_path} \
                --dataset ${dataset_name} \
                --split train \
                --val_size 0.05 \
                --cutoff_len 2048 \
                --preprocessing_num_workers 4 \
                --dataloader_num_workers 8 \
                --overwrite_cache \
                --evaluation_strategy steps \
                --eval_steps 100 \
                --template mistral \
                --finetuning_type full \
                --output_dir $model_path/$attacker_run_name/ \
                --overwrite_output_dir \
                --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
                --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
                --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
                --lr_scheduler_type cosine \
                --warmup_ratio 0.1 \
                --learning_rate $learning_rate \
                --num_train_epochs 1.0 \
                --top_k 0 \
                --top_p 0.9 \
                --logging_first_step \
                --logging_steps 5 \
                --run_name $STAGE-$attacker_run_name \
                --save_steps 1000 \
                --plot_loss \
                --bf16 \
                --seed 42 \
                --flash_attn fa2 \
            "
            echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
            printf "%b" "run_full_iteration.sh Iteration $iter, running command: Step $step: (attempt: $attempt): \"$CMD\"" | tee -a $log_file
            bash -c "$CMD"  2>&1 | tee -a $log_file
            
            cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
            if [ $cmd_status -eq 0 ]; then
                printf "%b" "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
                step=$((step+1))
                break
            else
                printf "%b" "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
                attempt=$((attempt+1))
                if [ $attempt -lt $RETRY_LIMIT ]; then
                    printf "%b" "Sleeping for $RETRY_DELAY seconds before retrying..." | tee -a $log_file
                    sleep $RETRY_DELAY
                else
                    printf "%b" "Reached maximum number of attempts. Exiting..." | tee -a $log_file
                    exit 1
                fi
            fi
        done

        # Step 0.4. Train the baseline defender model on adversarial-raw
        attempt=0
        until [ $attempt -ge $RETRY_LIMIT ]
        do
            if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
                printf "%b" "Skipping step $step on iter $iter"
                step=$((step+1))
                break
            fi
            STAGE=dpo
            model_name_or_path=$MISTRAL_MODEL
            baseline_run_name="auto-mistral_7b-$raw_dataset_name-$loss_type"

            TOTAL_BATCHSIZE=512
            PER_DEVICE_TRAIN_BATCH_SIZE=8
            GRADIENT_ACCUMULATION_STEPS=$(($TOTAL_BATCHSIZE / $NNODES / $GPUS_PER_NODE / $PER_DEVICE_TRAIN_BATCH_SIZE))
            # check if TOTAL_BATCHSIZE is divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE)
            if [ $(($GRADIENT_ACCUMULATION_STEPS * $NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE)) -ne $TOTAL_BATCHSIZE ]; then
                printf "%b" "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE} is not divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE): $(($NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE))" | tee -a $log_file
                exit 1
            fi
            printf "%b" "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE}, PER_DEVICE_TRAIN_BATCH_SIZE, ${PER_DEVICE_TRAIN_BATCH_SIZE}, GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}" | tee -a $log_file

            CMD="
            accelerate launch \
                --config_file ${ACCELERATE_CONFIG_PATH} \
                --main_process_ip ${MASTER_ADDR} \
                --main_process_port ${MASTER_PORT} \
                --num_processes ${NUM_PROCESSES} \
                --num_machines ${NNODES} \
                --machine_rank \$SLURM_PROCID \
            src/train.py \
                --stage $STAGE \
                --do_train \
                --do_eval \
                --model_name_or_path ${model_name_or_path} \
                --dataset $raw_dataset_name \
                --split train \
                --val_size 0.05 \
                --preprocessing_num_workers 4 \
                --dataloader_num_workers 8 \
                --overwrite_cache \
                --evaluation_strategy steps \
                --eval_steps 100 \
                --template mistral \
                --finetuning_type full \
                --output_dir $model_path/$baseline_run_name/ \
                --overwrite_output_dir \
                --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
                --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
                --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
                --lr_scheduler_type cosine \
                --warmup_ratio 0.1 \
                --learning_rate $learning_rate \
                --num_train_epochs 1.0 \
                --top_k 0 \
                --top_p 0.9 \
                --logging_first_step \
                --logging_steps 5 \
                --run_name $STAGE-$baseline_run_name \
                --save_steps 1000 \
                --plot_loss \
                --bf16 \
                --seed 42 \
                --flash_attn fa2 \
                --pref_loss $pref_loss \
                --pref_beta $pref_beta \
                --simpo_gamma $simpo_gamma \
                
            "
            echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
            printf "%b" "run_full_iteration.sh Iteration $iter, running command: Step $step: (attempt: $attempt): \"$CMD\"" | tee -a $log_file
            bash -c "$CMD"  2>&1 | tee -a $log_file
            
            cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
            if [ $cmd_status -eq 0 ]; then
                printf "%b" "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
                step=$((step+1))
                break
            else
                printf "%b" "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
                attempt=$((attempt+1))
                if [ $attempt -lt $RETRY_LIMIT ]; then
                    printf "%b" "Sleeping for $RETRY_DELAY seconds before retrying..." | tee -a $log_file
                    sleep $RETRY_DELAY
                else
                    printf "%b" "Reached maximum number of attempts. Exiting..." | tee -a $log_file
                    exit 1
                fi
            fi
        done

    else
        # iter >= 1
        mkdir -p $data_path/iterate-${iter}
        log_file="$data_path/iterate-${iter}/main.log"
        base_attacker_model=$model_path/auto-mistral_7b-$reverse_raw_dataset_name-$loss_type
        if [ $iter -eq 1 ]; then
            # reference_attacker_model=$MISTRAL_MODEL
            reference_attacker_model=$model_path/auto-mistral_7b-$reverse_raw_dataset_name-$loss_type
            reference_defender_model=$model_path/auto-mistral_7b-$raw_dataset_name-$loss_type
            reference_baseline_model=$model_path/auto-mistral_7b-$raw_dataset_name-$loss_type
        else
            prev_iter=$((iter - 1))
            reference_attacker_model=$model_path/auto-mistral_7b-adversarial-$prev_iter-attacker-$loss_type
            reference_defender_model=$model_path/auto-mistral_7b-adversarial-$prev_iter-$defender_dataset_suffix-$loss_type
            reference_baseline_model=$model_path/auto-mistral_7b-adversarial-$prev_iter-baseline-$loss_type
        fi
        printf "%b" "Running iteration $iter" | tee $log_file
        printf "%b" "Using reference_attacker_model: $reference_attacker_model" | tee -a $log_file
        printf "%b" "Using reference_defender_model: $reference_defender_model" | tee -a $log_file
        step=1

        # Step 1. Generate harmful prompts
        attempt=0
        until [ $attempt -ge $RETRY_LIMIT ]
        do
            if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
                printf "%b" "Skipping step $step on iter $iter"
                step=$((step+1))
                break
            fi
            CMD="
            python scripts/generate_prompts.py \
                --dataset_lists data/adversarial_dataset/adversarial_dataset.py adversarial-$iter-baseline \
                --model $reference_attacker_model \
                --tensor-parallel-size $GPUS_PER_NODE \
                --num_train_samples -1 \
                --num_test_samples -1 \
                --num_generated_prompts 8 \
                --prompt_ver $harmful_prompt_ver \
                --output_dir $data_path/iterate-${iter}/prompt \
                --disable-log-requests \
            "
            echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
            printf "%b" "run_full_iteration.sh Iteration $iter, running command: Step $step: (attempt: $attempt): \"$CMD\"" | tee -a $log_file
            bash -c "$CMD"  2>&1 | tee -a $log_file

            cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
            if [ $cmd_status -eq 0 ]; then
                printf "%b" "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
                step=$((step+1))
                break
            else
                printf "%b" "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
                attempt=$((attempt+1))
                if [ $attempt -lt $RETRY_LIMIT ]; then
                    printf "%b" "Sleeping for $RETRY_DELAY seconds before retrying..." | tee -a $log_file
                    sleep $RETRY_DELAY
                else
                    printf "%b" "Reached maximum number of attempts. Exiting..." | tee -a $log_file
                    exit 1
                fi
            fi
        done

        # Step 2. Calculate prompt rewards
        attempt=0
        until [ $attempt -ge $RETRY_LIMIT ]
        do
            if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
                printf "%b" "Skipping step $step on iter $iter"
                step=$((step+1))
                break
            fi
            CMD="
            accelerate launch \
                --multi_gpu \
                --config_file ${ACCELERATE_CONFIG_PATH} \
                --main_process_ip ${MASTER_ADDR} \
                --main_process_port ${MASTER_PORT} \
                --num_processes ${NUM_PROCESSES} \
                --num_machines ${NNODES} \
            scripts/calc_prompt_reward.py \
                --stage sft \
                --model_name_or_path $reference_defender_model \
                --raw_data_path $data_path/iterate-${iter}/prompt \
                --dataset_splits train test \
                --save_name $data_path/iterate-${iter}/score \
                --batch_size 8 \
                --preprocessing_num_workers 4 \
                --dataloader_num_workers 8 \
                --template mistral \
                --cutoff_len 2048 \
                --bf16 \
                --flash_attn off \
                --beta 1 \
            "
            echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
            printf "%b" "run_full_iteration.sh Iteration $iter, running command: Step $step: (attempt: $attempt): \"$CMD\"" | tee -a $log_file
            bash -c "$CMD"  2>&1 | tee -a $log_file

            cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
            if [ $cmd_status -eq 0 ]; then
                printf "%b" "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
                step=$((step+1))
                break
            else
                printf "%b" "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
                attempt=$((attempt+1))
                if [ $attempt -lt $RETRY_LIMIT ]; then
                    printf "%b" "Sleeping for $RETRY_DELAY seconds before retrying..." | tee -a $log_file
                    sleep $RETRY_DELAY
                else
                    printf "%b" "Reached maximum number of attempts. Exiting..." | tee -a $log_file
                    exit 1
                fi
            fi
        done

        # Step 3. Train the attacker model
        attempt=0
        until [ $attempt -ge $RETRY_LIMIT ]
        do
            if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
                printf "%b" "Skipping step $step on iter $iter"
                step=$((step+1))
                break
            fi
            STAGE=dpo
            model_name_or_path=$reference_attacker_model
            dataset_name=adversarial-$iter-attacker
            attacker_run_name="auto-mistral_7b-$dataset_name-$loss_type"

            TOTAL_BATCHSIZE=512
            PER_DEVICE_TRAIN_BATCH_SIZE=4
            GRADIENT_ACCUMULATION_STEPS=$(($TOTAL_BATCHSIZE / $NNODES / $GPUS_PER_NODE / $PER_DEVICE_TRAIN_BATCH_SIZE))
            # check if TOTAL_BATCHSIZE is divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE)
            if [ $(($GRADIENT_ACCUMULATION_STEPS * $NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE)) -ne $TOTAL_BATCHSIZE ]; then
                printf "%b" "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE} is not divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE): $(($NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE))" | tee -a $log_file
                exit 1
            fi
            printf "%b" "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE}, PER_DEVICE_TRAIN_BATCH_SIZE, ${PER_DEVICE_TRAIN_BATCH_SIZE}, GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}" | tee -a $log_file

            CMD="
            accelerate launch \
                --config_file ${ACCELERATE_CONFIG_PATH} \
                --main_process_ip ${MASTER_ADDR} \
                --main_process_port ${MASTER_PORT} \
                --num_processes ${NUM_PROCESSES} \
                --num_machines ${NNODES} \
                --machine_rank \$SLURM_PROCID \
            src/train.py \
                --stage $STAGE \
                --do_train \
                --do_eval \
                --model_name_or_path ${model_name_or_path} \
                --dataset ${dataset_name} \
                --split train \
                --val_size 0.05 \
                --cutoff_len 2048 \
                --preprocessing_num_workers 4 \
                --dataloader_num_workers 8 \
                --overwrite_cache \
                --evaluation_strategy steps \
                --eval_steps 100 \
                --template mistral \
                --finetuning_type full \
                --output_dir $model_path/$attacker_run_name/ \
                --overwrite_output_dir \
                --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
                --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
                --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
                --lr_scheduler_type cosine \
                --warmup_ratio 0.1 \
                --learning_rate $learning_rate \
                --num_train_epochs 1.0 \
                --top_k 0 \
                --top_p 0.9 \
                --logging_first_step \
                --logging_steps 5 \
                --run_name $STAGE-$attacker_run_name \
                --save_steps 1000 \
                --plot_loss \
                --bf16 \
                --seed 42 \
                --flash_attn fa2 \
                --pref_loss $pref_loss \
                --pref_beta $pref_beta \
                --simpo_gamma $simpo_gamma \
                --ref_model $base_attacker_model \
            "
            echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
            printf "%b" "run_full_iteration.sh Iteration $iter, running command: Step $step: (attempt: $attempt): \"$CMD\"" | tee -a $log_file
            bash -c "$CMD"  2>&1 | tee -a $log_file
            
            cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
            if [ $cmd_status -eq 0 ]; then
                printf "%b" "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
                step=$((step+1))
                break
            else
                printf "%b" "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
                attempt=$((attempt+1))
                if [ $attempt -lt $RETRY_LIMIT ]; then
                    printf "%b" "Sleeping for $RETRY_DELAY seconds before retrying..." | tee -a $log_file
                    sleep $RETRY_DELAY
                else
                    printf "%b" "Reached maximum number of attempts. Exiting..." | tee -a $log_file
                    exit 1
                fi
            fi
        done

        # Step 4. Train the defender model
        attempt=0
        until [ $attempt -ge $RETRY_LIMIT ]
        do
            if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
                printf "%b" "Skipping step $step on iter $iter"
                step=$((step+1))
                break
            fi
            STAGE=dpo
            model_name_or_path=$reference_defender_model
            dataset_name=adversarial-$iter-$defender_dataset_suffix
            defender_run_name="auto-mistral_7b-$dataset_name-$loss_type"

            TOTAL_BATCHSIZE=512
            PER_DEVICE_TRAIN_BATCH_SIZE=8
            GRADIENT_ACCUMULATION_STEPS=$(($TOTAL_BATCHSIZE / $NNODES / $GPUS_PER_NODE / $PER_DEVICE_TRAIN_BATCH_SIZE))
            # check if TOTAL_BATCHSIZE is divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE)
            if [ $(($GRADIENT_ACCUMULATION_STEPS * $NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE)) -ne $TOTAL_BATCHSIZE ]; then
                printf "%b" "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE} is not divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE): $(($NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE))" | tee -a $log_file
                exit 1
            fi
            printf "%b" "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE}, PER_DEVICE_TRAIN_BATCH_SIZE, ${PER_DEVICE_TRAIN_BATCH_SIZE}, GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}" | tee -a $log_file

            CMD="
            accelerate launch \
                --config_file ${ACCELERATE_CONFIG_PATH} \
                --main_process_ip ${MASTER_ADDR} \
                --main_process_port ${MASTER_PORT} \
                --num_processes ${NUM_PROCESSES} \
                --num_machines ${NNODES} \
                --machine_rank \$SLURM_PROCID \
            src/train.py \
                --stage $STAGE \
                --do_train \
                --do_eval \
                --model_name_or_path ${model_name_or_path} \
                --dataset ${dataset_name} \
                --split train \
                --val_size 0.05 \
                --preprocessing_num_workers 4 \
                --dataloader_num_workers 8 \
                --overwrite_cache \
                --evaluation_strategy steps \
                --eval_steps 100 \
                --template mistral \
                --finetuning_type full \
                --output_dir $model_path/$defender_run_name/ \
                --overwrite_output_dir \
                --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
                --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
                --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
                --lr_scheduler_type cosine \
                --warmup_ratio 0.1 \
                --learning_rate $learning_rate \
                --num_train_epochs 1.0 \
                --top_k 0 \
                --top_p 0.9 \
                --logging_first_step \
                --logging_steps 5 \
                --run_name $STAGE-$defender_run_name \
                --save_steps 1000 \
                --plot_loss \
                --bf16 \
                --seed 42 \
                --flash_attn fa2 \
                --pref_loss $pref_loss \
                --pref_beta $pref_beta \
                --simpo_gamma $simpo_gamma \
                
            "
            echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
            printf "%b" "run_full_iteration.sh Iteration $iter, running command: Step $step: (attempt: $attempt): \"$CMD\"" | tee -a $log_file
            bash -c "$CMD"  2>&1 | tee -a $log_file
            
            cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
            if [ $cmd_status -eq 0 ]; then
                printf "%b" "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
                step=$((step+1))
                break
            else
                printf "%b" "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
                attempt=$((attempt+1))
                if [ $attempt -lt $RETRY_LIMIT ]; then
                    printf "%b" "Sleeping for $RETRY_DELAY seconds before retrying..." | tee -a $log_file
                    sleep $RETRY_DELAY
                else
                    printf "%b" "Reached maximum number of attempts. Exiting..." | tee -a $log_file
                    exit 1
                fi
            fi
        done

        # Step 5. Train the baseline defender model
        attempt=0
        until [ $attempt -ge $RETRY_LIMIT ]
        do
            if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
                printf "%b" "Skipping step $step on iter $iter"
                step=$((step+1))
                break
            fi
            STAGE=dpo
            model_name_or_path=$reference_baseline_model
            dataset_name=adversarial-$iter-baseline
            baseline_run_name="auto-mistral_7b-$dataset_name-$loss_type"

            TOTAL_BATCHSIZE=512
            PER_DEVICE_TRAIN_BATCH_SIZE=8
            GRADIENT_ACCUMULATION_STEPS=$(($TOTAL_BATCHSIZE / $NNODES / $GPUS_PER_NODE / $PER_DEVICE_TRAIN_BATCH_SIZE))
            # check if TOTAL_BATCHSIZE is divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE)
            if [ $(($GRADIENT_ACCUMULATION_STEPS * $NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE)) -ne $TOTAL_BATCHSIZE ]; then
                printf "%b" "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE} is not divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE): $(($NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE))" | tee -a $log_file
                exit 1
            fi
            printf "%b" "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE}, PER_DEVICE_TRAIN_BATCH_SIZE, ${PER_DEVICE_TRAIN_BATCH_SIZE}, GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}" | tee -a $log_file

            CMD="
            accelerate launch \
                --config_file ${ACCELERATE_CONFIG_PATH} \
                --main_process_ip ${MASTER_ADDR} \
                --main_process_port ${MASTER_PORT} \
                --num_processes ${NUM_PROCESSES} \
                --num_machines ${NNODES} \
                --machine_rank \$SLURM_PROCID \
            src/train.py \
                --stage $STAGE \
                --do_train \
                --do_eval \
                --model_name_or_path ${model_name_or_path} \
                --dataset $dataset_name \
                --split train \
                --val_size 0.05 \
                --preprocessing_num_workers 4 \
                --dataloader_num_workers 8 \
                --overwrite_cache \
                --evaluation_strategy steps \
                --eval_steps 100 \
                --template mistral \
                --finetuning_type full \
                --output_dir $model_path/$baseline_run_name/ \
                --overwrite_output_dir \
                --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
                --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
                --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
                --lr_scheduler_type cosine \
                --warmup_ratio 0.1 \
                --learning_rate $learning_rate \
                --num_train_epochs 1.0 \
                --top_k 0 \
                --top_p 0.9 \
                --logging_first_step \
                --logging_steps 5 \
                --run_name $STAGE-$baseline_run_name \
                --save_steps 1000 \
                --plot_loss \
                --bf16 \
                --seed 42 \
                --flash_attn fa2 \
                --pref_loss $pref_loss \
                --pref_beta $pref_beta \
                --simpo_gamma $simpo_gamma \
                
            "
            echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
            printf "%b" "run_full_iteration.sh Iteration $iter, running command: Step $step: (attempt: $attempt): \"$CMD\"" | tee -a $log_file
            bash -c "$CMD"  2>&1 | tee -a $log_file
            
            cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
            if [ $cmd_status -eq 0 ]; then
                printf "%b" "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
                step=$((step+1))
                break
            else
                printf "%b" "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
                attempt=$((attempt+1))
                if [ $attempt -lt $RETRY_LIMIT ]; then
                    printf "%b" "Sleeping for $RETRY_DELAY seconds before retrying..." | tee -a $log_file
                    sleep $RETRY_DELAY
                else
                    printf "%b" "Reached maximum number of attempts. Exiting..." | tee -a $log_file
                    exit 1
                fi
            fi
        done
            
        reference_attacker_model=$model_path/$attacker_run_name
        reference_defender_model=$model_path/$defender_run_name
        reference_baseline_model=$model_path/$baseline_run_name
    fi
done

exit 0