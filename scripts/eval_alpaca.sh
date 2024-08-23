
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

alpaca_dataset_name=alpaca_eval
alpaca_dataset_name=hh_rlhf_harmless

pref_loss=sigmoid
defender_dataset_suffix=defender
start_iter=-1
num_iters=4
start_step=1
reference_attacker_model=$MISTRAL_MODEL
reference_defender_model=$ADV_MODEL_PATH/auto-mistral_7b-adversarial-raw-$pref_loss
baseline_defender_model=$ADV_MODEL_PATH/auto-mistral_7b-adversarial-raw-$pref_loss
data_path=$ADV_DATA_PATH
RETRY_LIMIT=3
RETRY_DELAY=60  # seconds

mkdir -p $data_path/eval/
log_file="$data_path/eval/alpaca_eval-$alpaca_dataset_name.log"
echo "Start evaluting for iteration $start_iter to $num_iters" | tee $log_file

for iter in $(seq $start_iter $num_iters); do
    echo "Running iteration $iter" | tee -a $log_file

    # Step 1. Generate alpaca response of defender models
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        if [ $iter -eq -1 ]; then
            model_name_or_path=$MISTRAL_MODEL
        elif [ $iter -eq 0 ]; then
            model_name_or_path=$reference_defender_model
        else
            model_name_or_path=$ADV_MODEL_PATH/auto-mistral_7b-adversarial-$iter-$defender_dataset_suffix-$pref_loss
        fi

        CMD="
        python scripts/generate_alpca_eval_responses.py \
            --model $model_name_or_path \
            --output_dir $data_path/eval/ \
            --disable-log-requests \
            --dataset_name $alpaca_dataset_name \
            # --max_samples 30 \
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

    # Step 2. Generate alpaca response of baseline methods
    attempt=0
    until [ $attempt -ge $RETRY_LIMIT ]
    do
        if [ $iter -ge 1 ]; then
            model_name_or_path=$ADV_MODEL_PATH/auto-mistral_7b-adversarial-$iter-baseline-$pref_loss
        else
            break
        fi

        CMD="
        python scripts/generate_alpca_eval_responses.py \
            --model $model_name_or_path \
            --output_dir $data_path/eval/ \
            --disable-log-requests \
            --dataset_name $alpaca_dataset_name \
            # --max_samples 30 \
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
done