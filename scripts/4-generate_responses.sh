
source scripts/.env
eval "$(conda shell.bash hook)"
conda activate llama-factory

declare -a urls=(
    "http://10.140.1.133:33998/v1"
)

declare -a dataset_names=(
    "iterate-0"
    # "iterate-1"
    # "iterate-2"
    # "iterate-3"
)
for dataset_name in "${dataset_names[@]}"; do

    # srun -p llm-safety --pty bash -c "
    unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY;
    python scripts/generate_responses.py \
        --endpoint_url ${urls[$rank]} \
        --model_name mistral \
        --raw_data_path data/adversarial_dataset/$dataset_name/prompt-mistral_7b/ \
        --output_dir data/adversarial_dataset/$dataset_name/eval-response-mistral_7b-adversarial-raw-sigmoid-adversarial-0-defender-sigmoid-adversarial-1-defender-sigmoid-adversarial-2-defender-sigmoid-adversarial-3-defender-sigmoid/ \
        --num_workers 48 \
        --num_prompts_per_example 1 \
        --num_generated_responses 1 \
        --num_max_retries 10 \
        --seed 42
    # "
    
done