
source scripts/.env
eval "$(conda shell.bash hook)"
conda activate adversarial-llm

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

data_path=$ADV_DATA_PATH
model_path=$ADV_MODEL_PATH

seed_dataset=relabeled_hh_rlhf_harmless
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

if [ ! -z "$1" ]; then
    start_iter=$1
    max_iters=$1
    start_step=1
else
    start_iter=0        # start from 0
    max_iters=30
    start_step=1
fi

loss_type=dpo
defender_dataset_suffix=defender
rm_model=output/rm/$seed_dataset/mistral_7b-adversarial-raw-1
# base_attacker_model=$model_path/auto-mistral_7b-adversarial-raw-reversed-$loss_type
base_attacker_model=$model_path/auto-mistral_7b-adversarial-harmless-reversed-$loss_type
base_defender_model=$model_path/auto-mistral_7b-adversarial-raw-$loss_type
base_baseline_model=$model_path/auto-mistral_7b-adversarial-raw-$loss_type
RETRY_LIMIT=3
RETRY_DELAY=60  # seconds

mkdir -p $data_path/eval/
log_file="$data_path/eval/main.log"
echo "Start evaluting for iteration $start_iter to $max_iters" | tee $log_file

for iter in $(seq $start_iter $max_iters); do
    if [ $iter -eq -1 ]; then
        continue
    fi
    echo "Running iteration $iter" | tee -a $log_file
    echo "Using base_attacker_model: $base_attacker_model" | tee -a $log_file
    echo "Using base_defender_model: $base_defender_model" | tee -a $log_file
    echo "Using base_baseline_model: $base_baseline_model" | tee -a $log_file
    echo "Using rm_model: $rm_model" | tee -a $log_file
    if [ $iter -eq 1 ]; then
        previous_attacker_model=$base_attacker_model
        previous_defender_model=$base_defender_model
        previous_baseline_model=$base_baseline_model
    elif [ $iter -gt 1 ]; then
        prev_iter=$((iter - 1))
        previous_attacker_model=$model_path/auto-mistral_7b-adversarial-$prev_iter-attacker-$loss_type
        previous_defender_model=$model_path/auto-mistral_7b-adversarial-$prev_iter-$defender_dataset_suffix-$loss_type
        previous_baseline_model=$model_path/auto-mistral_7b-adversarial-$prev_iter-baseline-$loss_type
    else
        previous_attacker_model=""
        previous_defender_model=""
        previous_baseline_model=""
    fi
    printf "%b" "Running iteration $iter" | tee $log_file
    printf "%b" "Using previous_attacker_model: $previous_attacker_model" | tee -a $log_file
    printf "%b" "Using previous_defender_model: $previous_defender_model" | tee -a $log_file
    step=1

    if [ $iter -eq 0 ]; then
        subfolder=base
    else
        subfolder=iterate-$iter
    fi

    # Step 1. Calculate base defender model's `chosen - rejected` score of `prompt-$subfolder`
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
            --model_name_or_path $base_defender_model \
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
        printf "%b" "eval_full_iteration.sh Iteration:: $iter (attempt: $attempt), running command: Step $step: $CMD" | tee -a $log_file
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

    # Step 2. Calculate previous defender model's `chosen - rejected` score of `prompt-$subfolder`
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        if [ $iter -eq 0 ]; then
            echo "Skipping step $step on iter $iter"
            step=$((step+1))
            break
        elif [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
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
            --model_name_or_path $previous_defender_model \
            --raw_data_path $data_path/$subfolder/prompt \
            --dataset_splits test \
            --save_name $data_path/eval/score-$subfolder-prev \
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
        printf "%b" "eval_full_iteration.sh Iteration:: $iter (attempt: $attempt), running command: Step $step: $CMD" | tee -a $log_file
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

    # Step 3. Generate base model responses to harmful prompts
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
            --model $base_defender_model \
            --tensor-parallel-size $GPUS_PER_NODE \
            --num_prompts_per_example 1 \
            --num_generated_responses 1 \
            --prompt_ver v1 \
            --output_dir $data_path/eval/prompt-$subfolder \
            --disable-log-requests \
        "
        echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
        printf "%b" "eval_full_iteration.sh Iteration:: $iter (attempt: $attempt), running command: Step $step: $CMD" | tee -a $log_file
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

    # Step 4. Calculate rewards of base model responses to harmful prompts
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
        printf "%b" "eval_full_iteration.sh Iteration:: $iter (attempt: $attempt), running command: Step $step: $CMD" | tee -a $log_file
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

    # Step 5. Generate defender responses to base harmful prompts
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
            echo "Skipping step $step on iter $iter"
            step=$((step+1))
            break
        fi
        if [ $iter -eq 0 ]; then
            model_name_or_path=$base_defender_model
            output_name=$data_path/eval/response-base-$defender_dataset_suffix
        else
            STAGE=dpo
            dataset_name=adversarial-$iter-$defender_dataset_suffix
            defender_run_name="auto-mistral_7b-$dataset_name-$loss_type"

            model_name_or_path=$model_path/$defender_run_name
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
        printf "%b" "eval_full_iteration.sh Iteration:: $iter (attempt: $attempt), running command: Step $step: $CMD" | tee -a $log_file
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

    # Step 6. Calculate rewards of defender responses to base harmful prompts
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
        printf "%b" "eval_full_iteration.sh Iteration:: $iter (attempt: $attempt), running command: Step $step: $CMD" | tee -a $log_file
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

    # Step 7. Generate baseline responses to base harmful prompts
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        if [ $step -lt $start_step ] && [ $iter -eq $start_iter ]; then
            echo "Skipping step $step on iter $iter"
            step=$((step+1))
            break
        fi
        if [ $iter -eq 0 ]; then
            model_name_or_path=$base_defender_model
            output_name=$data_path/eval/response-base-baseline
        else
            STAGE=dpo
            dataset_name=adversarial-$iter-baseline
            defender_run_name="auto-mistral_7b-$dataset_name-$loss_type"

            model_name_or_path=$model_path/$defender_run_name
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
        printf "%b" "eval_full_iteration.sh Iteration:: $iter (attempt: $attempt), running command: Step $step: $CMD" | tee -a $log_file
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

    # Step 8. Calculate rewards of baseline responses to base harmful prompts
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
        printf "%b" "eval_full_iteration.sh Iteration:: $iter (attempt: $attempt), running command: Step $step: $CMD" | tee -a $log_file
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
        # Step 9. Generate mistral responses to base harmful prompts
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
            printf "%b" "eval_full_iteration.sh Iteration:: $iter (attempt: $attempt), running command: Step $step: $CMD" | tee -a $log_file
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

        # Step 10. Calculate rewards of mistral responses to base harmful prompts
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
            printf "%b" "eval_full_iteration.sh Iteration:: $iter (attempt: $attempt), running command: Step $step: $CMD" | tee -a $log_file
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

exit 0