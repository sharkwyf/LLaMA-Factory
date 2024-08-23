
source scripts/.env
eval "$(conda shell.bash hook)"
conda activate llama-factory-bk

export CUDA_LAUNCH_BLOCKING=1
export ACCELERATE_LOG_LEVEL=info

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

seed_dataset=ultra_feedback
if [ $seed_dataset == "hh_rlhf_harmless" ]; then
    seed_dataset_path=data/custom_dataset/hh_rlhf.py
    seed_dataset_suffix=harmless
elif [ $seed_dataset == "ultra_feedback" ]; then
    seed_dataset_path=data/custom_dataset/ultra_feedback.py
    seed_dataset_suffix=all
fi

pref_loss=sigmoid
defender_dataset_suffix=defender
start_iter=0        # start from 0
num_iters=4
start_step=1
rm_model=output/rm/$seed_dataset/mistral_7b-adversarial-raw-1
reference_attacker_model=$MISTRAL_MODEL
reference_defender_model=$ADV_MODEL_PATH/auto-mistral_7b-adversarial-raw-$pref_loss
baseline_defender_model=$ADV_MODEL_PATH/auto-mistral_7b-adversarial-raw-$pref_loss
data_path=$ADV_DATA_PATH
RETRY_LIMIT=3
RETRY_DELAY=60  # seconds

mkdir -p $data_path/eval/
log_file="$data_path/eval/main.log"
echo "Start evaluting for iteration $start_iter to $num_iters" | tee $log_file

for iter in $(seq $start_iter $num_iters); do
    echo "Running iteration $iter" | tee -a $log_file
    echo "Using reference_attacker_model: $reference_attacker_model" | tee -a $log_file
    echo "Using reference_defender_model: $reference_defender_model" | tee -a $log_file
    echo "Using baseline_defender_model: $baseline_defender_model" | tee -a $log_file
    echo "Using rm_model: $rm_model" | tee -a $log_file
    step=1

    if [ $iter -eq 0 ]; then
        subfolder=base
    else
        subfolder=iterate-$iter
    fi

    # Step 1. Calculate chosen - rejected of harmful prompts
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
            echo "Skipping step $step on iter $iter"
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
            --raw_data_path $data_path/$subfolder/prompt \
            --dataset_splits test \
            --save_name $data_path/eval/score-$subfolder \
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
        echo "Iteration: $iter (attempt: $attempt), running command: Step 1: $CMD" | tee -a $log_file
        bash -c "$CMD"  2>&1 | tee -a $log_file

        cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
        if [ $cmd_status -eq 0 ]; then
            echo "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
            step=$((step+1))
            break
        else
            echo "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
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

    # Step 2. Generate base model responses to harmful prompts
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
            echo "Skipping step $step on iter $iter"
            step=$((step+1))
            break
        fi
        # TODO
        CMD="
        python scripts/generate_responses.py \
            --raw_data_path $data_path/$subfolder/prompt \
            --dataset_splits test \
            --model $reference_defender_model \
            --tensor-parallel-size $GPUS_PER_NODE \
            --num_prompts_per_example 1 \
            --num_generated_responses 1 \
            --prompt_ver v1 \
            --output_dir $data_path/eval/prompt-$subfolder \
            --disable-log-requests \
        "
        echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
        echo "Iteration: $iter (attempt: $attempt), running command: Step 2: $CMD" | tee -a $log_file
        bash -c "$CMD"  2>&1 | tee -a $log_file

        cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
        if [ $cmd_status -eq 0 ]; then
            echo "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
            step=$((step+1))
            break
        else
            echo "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
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

    # Step 3. Calculate rewards of base model responses to harmful prompts
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
            echo "Skipping step $step on iter $iter"
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
        scripts/calc_response_reward.py \
            --stage sft \
            --model_name_or_path $rm_model \
            --raw_data_path $data_path/eval/prompt-$subfolder \
            --dataset_splits test \
            --save_name $data_path/eval/prompt-$subfolder \
            --batch_size 8 \
            --preprocessing_num_workers 4 \
            --dataloader_num_workers 8 \
            --template mistral \
            --cutoff_len 2048 \
            --bf16 \
            --flash_attn off \
        "
        echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
        echo "Iteration: $iter (attempt: $attempt), running command: Step 3: $CMD" | tee -a $log_file
        bash -c "$CMD"  2>&1 | tee -a $log_file

        cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
        if [ $cmd_status -eq 0 ]; then
            echo "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
            step=$((step+1))
            break
        else
            echo "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
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

    # Step 4. Generate defender responses to base harmful prompts
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
            echo "Skipping step $step on iter $iter"
            step=$((step+1))
            break
        fi
        if [ $iter -eq 0 ]; then
            model_name_or_path=$reference_defender_model
            output_name=$data_path/eval/response-base-$defender_dataset_suffix
        else
            STAGE=dpo
            dataset_name=adversarial-$iter-$defender_dataset_suffix
            defender_run_name="auto-mistral_7b-$dataset_name-$pref_loss"

            model_name_or_path=$ADV_MODEL_PATH/$defender_run_name
            output_name=$data_path/eval/response-adversarial-$iter-$defender_dataset_suffix
        fi
        CMD="
        python scripts/generate_responses.py \
            --raw_data_path $data_path/base/prompt \
            --dataset_splits test \
            --model $model_name_or_path \
            --tensor-parallel-size $GPUS_PER_NODE \
            --num_prompts_per_example 1 \
            --num_generated_responses 1 \
            --prompt_ver v1 \
            --output_dir $output_name \
            --disable-log-requests \
        "
        echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
        echo "Iteration: $iter (attempt: $attempt), running command: Step 4: $CMD" | tee -a $log_file
        bash -c "$CMD"  2>&1 | tee -a $log_file

        cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
        if [ $cmd_status -eq 0 ]; then
            echo "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
            step=$((step+1))
            break
        else
            echo "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
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

    # Step 5. Calculate rewards of defender responses to base harmful prompts
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
            echo "Skipping step $step on iter $iter"
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
        scripts/calc_response_reward.py \
            --stage sft \
            --model_name_or_path $rm_model \
            --raw_data_path $output_name \
            --dataset_splits test \
            --save_name $output_name \
            --batch_size 8 \
            --preprocessing_num_workers 4 \
            --dataloader_num_workers 8 \
            --template mistral \
            --cutoff_len 2048 \
            --bf16 \
            --flash_attn off \
        "
        echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
        echo "Iteration: $iter (attempt: $attempt), running command: Step 5: $CMD" | tee -a $log_file
        bash -c "$CMD"  2>&1 | tee -a $log_file

        cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
        if [ $cmd_status -eq 0 ]; then
            echo "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
            step=$((step+1))
            break
        else
            echo "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
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

    # Step 6. Generate baseline responses to base harmful prompts
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
            echo "Skipping step $step on iter $iter"
            step=$((step+1))
            break
        fi
        if [ $iter -eq 0 ]; then
            model_name_or_path=$reference_defender_model
            output_name=$data_path/eval/response-base-baseline
        else
            STAGE=dpo
            dataset_name=adversarial-$iter-baseline
            defender_run_name="auto-mistral_7b-$dataset_name-$pref_loss"

            model_name_or_path=$ADV_MODEL_PATH/$defender_run_name
            output_name=$data_path/eval/response-adversarial-$iter-baseline
        fi
        CMD="
        python scripts/generate_responses.py \
            --raw_data_path $data_path/base/prompt \
            --dataset_splits test \
            --model $model_name_or_path \
            --tensor-parallel-size $GPUS_PER_NODE \
            --num_prompts_per_example 1 \
            --num_generated_responses 1 \
            --prompt_ver v1 \
            --output_dir $output_name \
            --disable-log-requests \
        "
        echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
        echo "Iteration: $iter (attempt: $attempt), running command: Step 6: $CMD" | tee -a $log_file
        bash -c "$CMD"  2>&1 | tee -a $log_file

        cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
        if [ $cmd_status -eq 0 ]; then
            echo "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
            step=$((step+1))
            break
        else
            echo "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
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

    # Step 7. Calculate rewards of baseline responses to base harmful prompts
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
            echo "Skipping step $step on iter $iter"
            step=$((step+1))
            break
        fi
        # TODO
        CMD="
        accelerate launch \
            --multi_gpu \
            --config_file ${ACCELERATE_CONFIG_PATH} \
            --main_process_ip ${MASTER_ADDR} \
            --main_process_port ${MASTER_PORT} \
            --num_processes ${NUM_PROCESSES} \
            --num_machines ${NNODES} \
        scripts/calc_response_reward.py \
            --stage sft \
            --model_name_or_path $rm_model \
            --raw_data_path $output_name \
            --dataset_splits test \
            --save_name $output_name \
            --batch_size 8 \
            --preprocessing_num_workers 4 \
            --dataloader_num_workers 8 \
            --template mistral \
            --cutoff_len 2048 \
            --bf16 \
            --flash_attn off \
        "
        echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
        echo "Iteration: $iter (attempt: $attempt), running command: Step 7: $CMD" | tee -a $log_file
        bash -c "$CMD"  2>&1 | tee -a $log_file

        cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
        if [ $cmd_status -eq 0 ]; then
            echo "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
            step=$((step+1))
            break
        else
            echo "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
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

    if [ $iter -eq 0 ]; then
        # Step E1. Generate mistral responses to base harmful prompts
        attempt=0
        until [ $attempt -ge $RETRY_LIMIT ]
        do
            if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
                echo "Skipping step $step on iter $iter"
                step=$((step+1))
                break
            fi
            model_name_or_path=$MISTRAL_MODEL
            output_name=$data_path/eval/response-mistral
            CMD="
            python scripts/generate_responses.py \
                --raw_data_path $data_path/base/prompt \
                --dataset_splits test \
                --model $model_name_or_path \
                --tensor-parallel-size $GPUS_PER_NODE \
                --num_prompts_per_example 1 \
                --num_generated_responses 1 \
                --prompt_ver v1 \
                --output_dir $output_name \
                --disable-log-requests \
            "
            echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
            echo "Iteration: $iter (attempt: $attempt), running command: Step E1: $CMD" | tee -a $log_file
            bash -c "$CMD"  2>&1 | tee -a $log_file

            cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
            if [ $cmd_status -eq 0 ]; then
                echo "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
                step=$((step+1))
                break
            else
                echo "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
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

        # Step E2. Calculate rewards of mistral responses to base harmful prompts
        attempt=0
        until [ $attempt -ge $RETRY_LIMIT ]
        do
            if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
                echo "Skipping step $step on iter $iter"
                step=$((step+1))
                break
            fi
            # TODO
            CMD="
            accelerate launch \
                --multi_gpu \
                --config_file ${ACCELERATE_CONFIG_PATH} \
                --main_process_ip ${MASTER_ADDR} \
                --main_process_port ${MASTER_PORT} \
                --num_processes ${NUM_PROCESSES} \
                --num_machines ${NNODES} \
            scripts/calc_response_reward.py \
                --stage sft \
                --model_name_or_path $rm_model \
                --raw_data_path $output_name \
                --dataset_splits test \
                --save_name $output_name \
                --batch_size 8 \
                --preprocessing_num_workers 4 \
                --dataloader_num_workers 8 \
                --template mistral \
                --cutoff_len 2048 \
                --bf16 \
                --flash_attn off \
            "
            echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
            echo "Iteration: $iter (attempt: $attempt), running command: Step E2: $CMD" | tee -a $log_file
            bash -c "$CMD"  2>&1 | tee -a $log_file

            cmd_status=${PIPESTATUS[0]}  # Get the exit status of the bash command, not tee
            if [ $cmd_status -eq 0 ]; then
                echo "Command succeeded on iter $iter step $step attempt $attempt" | tee -a $log_file
                step=$((step+1))
                break
            else
                echo "Command failed on iter $iter step $step attempt $attempt with status $cmd_status" | tee -a $log_file
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
        
    else
        continue
    fi
done