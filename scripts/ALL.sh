
source scripts/.env
eval "$(conda shell.bash hook)"
conda activate llama-factory-bk

data_path=$ADV_DATA_PATH
model_path=$ADV_MODEL_PATH

start_iter=-1       # start from -1
max_iters=30

for iter in $(seq $start_iter $max_iters); do
    bash scripts/run_full_iteration.sh $iter $iter
    if [ $? -ne 0 ]; then
        echo "run_full_iteration.sh failed with exit status $?"
        exit 1
    fi

    bash scripts/eval_full_iteration.sh $iter $iter
    if [ $? -ne 0 ]; then
        echo "eval_full_iteration.sh failed with exit status $?"
        exit 1
    fi

    bash scripts/eval_alpaca_harmless.sh $iter $iter
    if [ $? -ne 0 ]; then
        echo "eval_alpaca_harmless.sh failed with exit status $?"
        exit 1
    fi

    bash scripts/eval_alpaca_helpfulness.sh $iter $iter
    if [ $? -ne 0 ]; then
        echo "eval_alpaca_helpfulness.sh failed with exit status $?"
        exit 1
    fi

    if [ $iter -ge 2 ]; then
        prev_iter=$((iter-1))
        if [ $((prev_iter % 5)) -ne 0 ]; then
                rm -rf $model_path/auto-*-adversarial-$prev_iter-*
        fi
    fi
done
