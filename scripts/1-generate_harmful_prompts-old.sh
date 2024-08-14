
source scripts/.env
eval "$(conda shell.bash hook)"
conda activate llama-factory

# if $1 is not set, set it to 0
if [ -z "$1" ]
then
    rank=0
else
    rank=$1
fi
num_worlds=1

declare -a urls=(
    "http://10.140.1.133:33998/v1"
)


dataset_name=iterate-3
# srun -p llm-safety --pty bash -c "
unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY;
python scripts/generate_harmful_prompts_old.py \
    --endpoint_url ${urls[$rank]} \
    --output_dir data/adversarial_dataset/${dataset_name}/prompt-mistral_7b-adversarial-raw-sigmoid-adversarial-raw-sigmoid-adversarial-raw-sigmoid-adversarial-raw-sigmoid/ \
    --num_workers 32 \
    --num_worlds $num_worlds \
    --num_train_samples -1 \
    --num_test_samples -1 \
    --num_generated_prompts 16 \
    --rank $rank
# "