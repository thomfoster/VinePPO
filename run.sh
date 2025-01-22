

# VINEPPO
# CONFIGSTR="configs/polIter_deepseekSft2_vineppo_MATH.jsonnet,\
# CONFIGSTR="configs/polIter_rho1bSft2_vineppo_MATH.jsonnet,\
# configs/trainers/devBz16.jsonnet"
CONFIGSTR="configs/polIter_rho1bSft2_vineppo_MATH.jsonnet,\
configs/trainers/devBz16.jsonnet,\
configs/adjust_temp_dir.jsonnet"

# VINEPPO - GSM8K
# CONFIGSTR="configs/polIter_deepseekSft2_vineppo_MATH.jsonnet,\
# CONFIGSTR="configs/polIter_rho1bSft2_vineppo_GSM8K.jsonnet,\
# configs/trainers/devBz16.jsonnet"
# CONFIGSTR="configs/polIter_rho1bSft2_vineppo_GSM8K.jsonnet,\
# configs/trainers/devBz16.jsonnet,\
# configs/adjust_temp_dir_for_gsm8k_no_sfl.jsonnet"

# PPO 
# CONFIGSTR="configs/polIter_rho1bSft2_ppo_MATH.jsonnet,\
# configs/trainers/devBz16.jsonnet"

APP_DIRECTORY="experiments/"

export APP_SEED="2746318213"
# export WANDB_RUN_ID="<unique_wandb_run_id>" # Optional

export HF_TOKEN="XXX"
export WANDB_API_KEY="XXX"

#NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_GPUS=1

# Run the training
deepspeed --no_local_rank --num_gpus=$NUM_GPUS  \
         src/treetune/main.py --configs "$CONFIGSTR" \
            run_iteration_loop

# Run the evaluation
# deepspeed --no_local_rank --num_gpus=1   \
#          src/treetune/main.py --configs "$CONFIGSTR" \
#             run_evaluation

