
source scripts/.env
eval "$(conda shell.bash hook)"
conda activate llama-factory

IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPUS[@]}
echo "Number of GPUs: ${#GPUS[@]}"

model_name_or_path=output/dpo/mistral_7b-adversarial-raw-sigmoid-adversarial-0-defender-sigmoid-adversarial-1-defender-sigmoid-adversarial-2-defender-sigmoid
dataset_name=iterate-3
model_base_name=$(echo $model_name_or_path | awk -F'/' '{print $NF}')
unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY;
python scripts/generate_harmful_prompts.py \
    --dataset_lists data/custom_dataset/hh_rlhf.py harmless \
    --model $model_name_or_path \
    --tensor-parallel-size $NUM_GPUS \
    --num_train_samples -1 \
    --num_test_samples -1 \
    --num_generated_prompts 16 \
    --prompt_ver v1 \
    --output_dir data/adversarial_dataset/${dataset_name}/prompt-$model_base_name/ \