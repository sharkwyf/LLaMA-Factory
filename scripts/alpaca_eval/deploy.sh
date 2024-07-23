

IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPUS[@]}
echo "Number of GPUs: $NUM_GPUS"

model_type=$1

if [ "$model_type" == "internlm2" ]; then
  model_name_or_path=/mnt/hwfile/llm-safety/models/internlm2-chat-20b
  served_model_name=gpt-4-1106-preview

elif [ "$model_type" == "qwen2" ]; then
  model_name_or_path=/mnt/hwfile/llm-safety/models/Qwen2-72B-Instruct
  served_model_name=gpt-4-1106-preview
  served_model_name=gpt-3.5-turbo-1106
  served_model_name=qwen2

else
    echo "Invalid model type"
    exit 1
fi

echo "Deploying model $served_model_name from $model_name_or_path"

# if SLURM_JOBID exists in env
if [ -n "$SLURM_JOBID" ]; then
  echo "Running via bash"
  python -m vllm.entrypoints.openai.api_server \
    --model $model_name_or_path \
    --served-model-name $served_model_name \
    --host 0.0.0.0 \
    --port 33998 \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size $NUM_GPUS

else
  echo "Running via srun"
  srun -p llm-safety --gres=gpu:$NUM_GPUS -J vllm \
  python -m vllm.entrypoints.openai.api_server \
    --model $model_name_or_path \
    --served-model-name $served_model_name \
    --host 0.0.0.0 \
    --port 33998 \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size $NUM_GPUS
    
fi

  
