
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
echo "Number of GPUs: ${#GPUS[@]}"
GPUS_PER_NODE=${#GPUS[@]}
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

ACCELERATE_CONFIG_PATH=scripts/accelerate_config.yaml
DS_CONFIG_PATH=scripts/ds_config.json

defender_dataset_suffix=defender-mixed
start_iter=1
num_iters=1
if [ $start_iter -eq 1 ]; then
    reference_attacker_model=$MISTRAL_MODEL
    reference_defender_model=output/dpo/mistral_7b-adversarial-raw-sigmoid
    baseline_defender_model=output/dpo/mistral_7b-adversarial-raw-sigmoid
else
    prev_iter=$((start_iter-1))
    reference_attacker_model=output/dpo/auto-mistral_7b-adversarial-$prev_iter-attacker-sigmoid
    reference_defender_model=output/dpo/auto-mistral_7b-adversarial-$prev_iter-$defender_dataset_suffix-sigmoid
    baseline_defender_model=output/dpo/auto-mistral_7b-adversarial-$prev_iter-baseline-sigmoid
fi
data_path=$ADV_DATA_PATH
RETRY_LIMIT=3
RETRY_DELAY=60  # seconds

for iter in $(seq $start_iter $num_iters); do
    mkdir -p $data_path/iterate-${iter}
    log_file="$data_path/iterate-${iter}/main.log"
    echo "Running iteration $iter" | tee $log_file
    echo "Using reference_attacker_model: $reference_attacker_model" | tee -a $log_file
    echo "Using reference_defender_model: $reference_defender_model" | tee -a $log_file

    # Step 1. Generate harmful prompts
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        CMD="
        python scripts/generate_harmful_prompts.py \
            --dataset_lists data/custom_dataset/hh_rlhf.py harmless \
            --model $reference_attacker_model \
            --tensor-parallel-size $GPUS_PER_NODE \
            --num_train_samples -1 \
            --num_test_samples -1 \
            --num_generated_prompts 8 \
            --prompt_ver v1 \
            --output_dir $data_path/iterate-${iter}/prompt \
            --disable-log-requests \
        "
        echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
        echo "Iteration: $iter (attempt: $attempt), running command: Step 1: $CMD" | tee -a $log_file
        bash -c "$CMD"  2>&1 | tee -a $log_file

        cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
        if [ $cmd_status -eq 0 ]; then
            echo "Command succeeded on attempt $attempt" | tee -a $log_file
            break
        else
            echo "Command failed with status $cmd_status" | tee -a $log_file
            attempt=$((attempt+1))
            if [ $attempt -lt $RETRY_LIMIT ]; then
                echo "Sleeping for $RETRY_DELAY seconds before retrying..." | tee -a $log_file
                sleep $RETRY_DELAY
            else
                echo "Reached maximum number of attempts. Exiting..." | tee -a $log_file
                exit 1
            fi
        fi
    done

    # Step 2. Calculate prompt rewards
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
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
        echo "Iteration: $iter (attempt: $attempt), running command: Step 2: $CMD" | tee -a $log_file
        bash -c "$CMD"  2>&1 | tee -a $log_file

        cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
        if [ $cmd_status -eq 0 ]; then
            echo "Command succeeded on attempt $attempt" | tee -a $log_file
            break
        else
            echo "Command failed with status $cmd_status" | tee -a $log_file
            attempt=$((attempt+1))
            if [ $attempt -lt $RETRY_LIMIT ]; then
                echo "Sleeping for $RETRY_DELAY seconds before retrying..." | tee -a $log_file
                sleep $RETRY_DELAY
            else
                echo "Reached maximum number of attempts. Exiting..." | tee -a $log_file
                exit 1
            fi
        fi
    done

    # Step 3. Train the attacker model
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        STAGE=dpo
        model_name_or_path=$reference_attacker_model
        dataset_name=adversarial-$iter-attacker
        pref_loss=sigmoid
        pref_beta=0.01
        attacker_run_name="auto-mistral_7b-$dataset_name-$pref_loss"

        TOTAL_BATCHSIZE=512
        PER_DEVICE_TRAIN_BATCH_SIZE=4
        GRADIENT_ACCUMULATION_STEPS=$(($TOTAL_BATCHSIZE / $NNODES / $GPUS_PER_NODE / $PER_DEVICE_TRAIN_BATCH_SIZE))
        # check if TOTAL_BATCHSIZE is divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE)
        if [ $(($GRADIENT_ACCUMULATION_STEPS * $NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE)) -ne $TOTAL_BATCHSIZE ]; then
            echo "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE} is not divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE): $(($NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE))" | tee -a $log_file
            exit 1
        fi
        echo "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE}, PER_DEVICE_TRAIN_BATCH_SIZE, ${PER_DEVICE_TRAIN_BATCH_SIZE}, GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}" | tee -a $log_file

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
            --output_dir output/$STAGE/$attacker_run_name/ \
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
            --run_name $STAGE-$attacker_run_name \
            --save_steps 1000 \
            --plot_loss \
            --bf16 \
            --seed 42 \
            --flash_attn fa2 \
            --pref_loss $pref_loss \
            --pref_beta $pref_beta \
        "
        echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
        echo "Iteration: $iter (attempt: $attempt), running command: Step 3: $CMD" | tee -a $log_file
        bash -c "$CMD"  2>&1 | tee -a $log_file
        
        cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
        if [ $cmd_status -eq 0 ]; then
            echo "Command succeeded on attempt $attempt" | tee -a $log_file
            break
        else
            echo "Command failed with status $cmd_status" | tee -a $log_file
            attempt=$((attempt+1))
            if [ $attempt -lt $RETRY_LIMIT ]; then
                echo "Sleeping for $RETRY_DELAY seconds before retrying..." | tee -a $log_file
                sleep $RETRY_DELAY
            else
                echo "Reached maximum number of attempts. Exiting..." | tee -a $log_file
                exit 1
            fi
        fi
    done

    # Step 4. Train the defender model
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        STAGE=dpo
        model_name_or_path=$reference_defender_model
        dataset_name=adversarial-$iter-$defender_dataset_suffix
        pref_loss=sigmoid
        pref_beta=0.01
        defender_run_name="auto-mistral_7b-$dataset_name-$pref_loss"

        TOTAL_BATCHSIZE=512
        PER_DEVICE_TRAIN_BATCH_SIZE=8
        GRADIENT_ACCUMULATION_STEPS=$(($TOTAL_BATCHSIZE / $NNODES / $GPUS_PER_NODE / $PER_DEVICE_TRAIN_BATCH_SIZE))
        # check if TOTAL_BATCHSIZE is divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE)
        if [ $(($GRADIENT_ACCUMULATION_STEPS * $NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE)) -ne $TOTAL_BATCHSIZE ]; then
            echo "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE} is not divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE): $(($NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE))" | tee -a $log_file
            exit 1
        fi
        echo "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE}, PER_DEVICE_TRAIN_BATCH_SIZE, ${PER_DEVICE_TRAIN_BATCH_SIZE}, GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}" | tee -a $log_file

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
            --output_dir output/$STAGE/$defender_run_name/ \
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
            --run_name $STAGE-$defender_run_name \
            --save_steps 1000 \
            --plot_loss \
            --bf16 \
            --seed 42 \
            --flash_attn fa2 \
            --pref_loss $pref_loss \
            --pref_beta $pref_beta \
            
        "
        echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
        echo "Iteration: $iter (attempt: $attempt), running command: Step 4: $CMD"
        bash -c "$CMD"  2>&1 | tee -a $log_file
        
        cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
        if [ $cmd_status -eq 0 ]; then
            echo "Command succeeded on attempt $attempt" | tee -a $log_file
            break
        else
            echo "Command failed with status $cmd_status" | tee -a $log_file
            attempt=$((attempt+1))
            if [ $attempt -lt $RETRY_LIMIT ]; then
                echo "Sleeping for $RETRY_DELAY seconds before retrying..." | tee -a $log_file
                sleep $RETRY_DELAY
            else
                echo "Reached maximum number of attempts. Exiting..." | tee -a $log_file
                exit 1
            fi
        fi
    done

    # Step 5. Train the baseline defender model
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        STAGE=dpo
        model_name_or_path=$baseline_defender_model
        dataset_name=adversarial-$iter-baseline
        pref_loss=sigmoid
        pref_beta=0.01
        baseline_run_name="auto-mistral_7b-$dataset_name-$pref_loss"

        TOTAL_BATCHSIZE=512
        PER_DEVICE_TRAIN_BATCH_SIZE=8
        GRADIENT_ACCUMULATION_STEPS=$(($TOTAL_BATCHSIZE / $NNODES / $GPUS_PER_NODE / $PER_DEVICE_TRAIN_BATCH_SIZE))
        # check if TOTAL_BATCHSIZE is divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE)
        if [ $(($GRADIENT_ACCUMULATION_STEPS * $NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE)) -ne $TOTAL_BATCHSIZE ]; then
            echo "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE} is not divisible by (NNODES * GPUS_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE): $(($NNODES * $GPUS_PER_NODE * $PER_DEVICE_TRAIN_BATCH_SIZE))" | tee -a $log_file
            exit 1
        fi
        echo "TOTAL_BATCHSIZE: ${TOTAL_BATCHSIZE}, PER_DEVICE_TRAIN_BATCH_SIZE, ${PER_DEVICE_TRAIN_BATCH_SIZE}, GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}" | tee -a $log_file

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
            --dataset adversarial-raw \
            --split train \
            --val_size 0.05 \
            --preprocessing_num_workers 4 \
            --dataloader_num_workers 8 \
            --overwrite_cache \
            --evaluation_strategy steps \
            --eval_steps 100 \
            --template mistral \
            --finetuning_type full \
            --output_dir output/$STAGE/$baseline_run_name/ \
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
            --run_name $STAGE-$baseline_run_name \
            --save_steps 1000 \
            --plot_loss \
            --bf16 \
            --seed 42 \
            --flash_attn fa2 \
            --pref_loss $pref_loss \
            --pref_beta $pref_beta \
            
        "
        echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
        echo "Iteration: $iter (attempt: $attempt), running command: Step 4: $CMD"
        bash -c "$CMD"  2>&1 | tee -a $log_file
        
        cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
        if [ $cmd_status -eq 0 ]; then
            echo "Command succeeded on attempt $attempt" | tee -a $log_file
            break
        else
            echo "Command failed with status $cmd_status" | tee -a $log_file
            attempt=$((attempt+1))
            if [ $attempt -lt $RETRY_LIMIT ]; then
                echo "Sleeping for $RETRY_DELAY seconds before retrying..." | tee -a $log_file
                sleep $RETRY_DELAY
            else
                echo "Reached maximum number of attempts. Exiting..." | tee -a $log_file
                exit 1
            fi
        fi
    done
        
    reference_attacker_model=output/$STAGE/$attacker_run_name
    reference_defender_model=output/$STAGE/$defender_run_name
    baseline_defender_model=output/$STAGE/$baseline_run_name

done