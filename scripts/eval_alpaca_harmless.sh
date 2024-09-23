
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

data_path=$ADV_DATA_PATH
model_path=$ADV_MODEL_PATH

alpaca_dataset_name=hh_rlhf_harmless

if [ ! -z "$1" ]; then
    start_iter=$1
    max_iters=$1
    start_step=1
else
    start_iter=-1        # start from 0
    max_iters=30
    start_step=1
fi

loss_type=dpo
defender_dataset_suffix=defender
# base_attacker_model=$model_path/auto-mistral_7b-adversarial-raw-reversed-$loss_type
base_attacker_model=$model_path/auto-mistral_7b-adversarial-harmless-reversed-$loss_type
base_defender_model=$model_path/auto-mistral_7b-adversarial-raw-$loss_type
base_baseline_model=$model_path/auto-mistral_7b-adversarial-raw-$loss_type
RETRY_LIMIT=3
RETRY_DELAY=60  # seconds

mkdir -p $data_path/eval/
log_file="$data_path/eval/alpaca_eval-$alpaca_dataset_name.log"
echo "Start evaluting for iteration $start_iter to $max_iters" | tee $log_file

for iter in $(seq $start_iter $max_iters); do
    echo "Running iteration $iter" | tee -a $log_file

    # Step 0. Convert generated prompts to alpaca format for base (Run Once)
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        if [ $iter -eq 1 ]; then
            subfolder=iterate-1
            as_base=true
        else
            break
        fi

        CMD="
        python scripts/convert_alpaca_format.py \
            --raw_data_path $data_path/$subfolder/prompt/test.json \
            --dataset_name $alpaca_dataset_name \
            --test_samples_per_iter -1 \
            --output_dir $data_path/eval/ \
        "
        if [ $as_base = true ]; then
            CMD+="--as_base "
        fi
        echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
        printf "%b" "eval_alpaca.sh Iteration:: $iter (attempt: $attempt), running command: Step 0: $CMD" | tee -a $log_file
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

    # Step 1. Convert generated prompts to alpaca format
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        if [ $iter -le 0 ]; then
            break
        else
            subfolder=iterate-$iter
            as_base=false
        fi

        CMD="
        python scripts/convert_alpaca_format.py \
            --raw_data_path $data_path/$subfolder/prompt/test.json \
            --dataset_name $alpaca_dataset_name \
            --test_samples_per_iter -1 \
            --output_dir $data_path/eval/ \
        "
        if [ $as_base = true ]; then
            CMD += "--as_base "
        fi
        echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
        printf "%b" "eval_alpaca.sh Iteration:: $iter (attempt: $attempt), running command: Step 1: $CMD" | tee -a $log_file
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

    # Step 2. Generate alpaca response of defender models
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        if [ $iter -eq -1 ]; then
            model_name_or_path=$MISTRAL_MODEL
        elif [ $iter -eq 0 ]; then
            model_name_or_path=$base_defender_model
        else
            model_name_or_path=$model_path/auto-mistral_7b-adversarial-$iter-$defender_dataset_suffix-$loss_type
        fi

        CMD="
        python scripts/generate_alpca_eval_responses.py \
            --model $model_name_or_path \
            --output_dir $data_path/eval/ \
            --disable-log-requests \
            --dataset_name $alpaca_dataset_name \
        "
        echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
        printf "%b" "eval_alpaca.sh Iteration:: $iter (attempt: $attempt), running command: Step 2: $CMD" | tee -a $log_file
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

    # Step 3. Generate alpaca response of baseline methods
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        if [ $iter -ge 1 ]; then
            model_name_or_path=$model_path/auto-mistral_7b-adversarial-$iter-baseline-$loss_type
        else
            break
        fi

        CMD="
        python scripts/generate_alpca_eval_responses.py \
            --model $model_name_or_path \
            --output_dir $data_path/eval/ \
            --disable-log-requests \
            --dataset_name $alpaca_dataset_name \
        "
        echo -e "\n\n\n======================================================\n\n\n" | tee -a $log_file
        printf "%b" "eval_alpaca.sh Iteration:: $iter (attempt: $attempt), running command: Step 3: $CMD" | tee -a $log_file
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
done

exit 0