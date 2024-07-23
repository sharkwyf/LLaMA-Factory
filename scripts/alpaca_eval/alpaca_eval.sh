unset OPENAI_CLIENT_CONFIG_PATH
unset OPENAI_API_BASE
unset OPENAI_API_KEY

# local
# export OPENAI_CLIENT_CONFIG_PATH="/mnt/petrelfs/wangyuanfu/workspace/AdversarialLLM/scripts/alpaca_eval/openai_configs_example.yaml"
# export annotators_config=/mnt/petrelfs/wangyuanfu/workspace/AdversarialLLM/scripts/alpaca_eval/alpaca_eval_qwen2_fn

# gpt4_turbo
export annotators_config=/mnt/petrelfs/wangyuanfu/workspace/AdversarialLLM/scripts/alpaca_eval/alpaca_eval_gpt4_turbo_fn
export http_proxy=http://closeai-proxy.pjlab.org.cn:23128
export https_proxy=http://closeai-proxy.pjlab.org.cn:23128
export HTTP_PROXY=http://closeai-proxy.pjlab.org.cn:23128
export HTTPS_PROXY=http://closeai-proxy.pjlab.org.cn:23128
curl -x "http://10.1.20.57:23128" ipinfo.io
export OPENAI_API_KEY="sk-3"
export OPENAI_PROXY_URL=http://closeai-proxy.pjlab.org.cn:23128

alpaca_eval evaluate \
    --reference_outputs data/adversarial_dataset/exp/eval/alpaca_eval_mistral_7b-adversarial-raw-sigmoid.json \
    --model_outputs data/adversarial_dataset/exp/eval/alpaca_eval_auto-mistral_7b-adversarial-1-defender-sigmoid.json \
    --name example_model \
    --annotators_config $annotators_config \
    --output_path scripts/alpaca_eval/output \
    --is_overwrite_leaderboard true \
    --max_instances 1 \

    # --reference_outputs alpaca_eval_gpt4_baseline \
    
