
source scripts/.env
eval "$(conda shell.bash hook)"
conda activate llama-factory-bk

unset OPENAI_CLIENT_CONFIG_PATH
unset OPENAI_API_BASE
unset OPENAI_API_KEY

# local
# export OPENAI_CLIENT_CONFIG_PATH="/mnt/petrelfs/wangyuanfu/workspace/AdversarialLLM/scripts/alpaca_eval/openai_configs_example.yaml"
# export annotators_config=/mnt/petrelfs/wangyuanfu/workspace/AdversarialLLM/scripts/alpaca_eval/alpaca_eval_qwen2_fn

# gpt4_turbo

export OPENAI_API_BASE="http://47.88.65.188:8300/v1"
export OPENAI_API_KEY="sk-plRSdCGsEhjUsZhA16D22444D39b42E2A41cB15982B99377"
# export OPENAI_PROXY_URL=http://closeai-proxy.pjlab.org.cn:23128
export data_path=$ADV_DATA_PATH
# export data_path=data/adversarial_dataset/hh_rlhf_multi-5-2

export alpaca_dataset_name=alpaca_eval

loss_type=dpo

mkdir -p $data_path/eval/$alpaca_dataset_name/
log_file="$data_path/eval/$alpaca_dataset_name/main.log"

reference_outputs=$data_path/eval/${alpaca_dataset_name}_Mistral-7B-Instruct-v0.2.json
annotators_config=/mnt/lustre/wangyuanfu/petrelfs/workspace/AdversarialLLM/scripts/alpaca_eval/${alpaca_dataset_name}_gpt4_turbo_fn

declare -a all_model_outputs=(
    $reference_outputs
    $data_path/eval/${alpaca_dataset_name}_auto-mistral_7b-adversarial-raw-${loss_type}.json
)

# Outer loop for 'baseline' and 'defender'
for model_type in baseline defender; do
    # Inner loop for different indices (raw and even numbers)
    for i in {5..30..5}; do
        all_model_outputs+=("$data_path/eval/${alpaca_dataset_name}_auto-mistral_7b-adversarial-$i-${model_type}-${loss_type}.json")
        echo "$data_path/eval/${alpaca_dataset_name}_auto-mistral_7b-adversarial-$i-${model_type}-${loss_type}.json"
    done
done

for model_outputs in "${all_model_outputs[@]}"; do
    echo "Evaluate on $model_outputs" 2>&1 | tee -a $log_file
    model_base_name=$(basename "$model_outputs" ".json")
    
    CMD="
    alpaca_eval evaluate \
        --reference_outputs $reference_outputs \
        --model_outputs $model_outputs \
        --name $model_base_name \
        --annotators_config $annotators_config \
        --precomputed_leaderboard $data_path/eval/${alpaca_dataset_name}/leaderboard.csv \
        --output_path $data_path/eval/${alpaca_dataset_name} \
        --is_overwrite_leaderboard true \
        --max_instances 60 \
    "

    bash -c "$CMD" 2>&1 | tee -a $log_file
done