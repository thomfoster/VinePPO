# export CUDA_VISIBLE_DEVICES=1,2,3,6
export CUDA_VISIBLE_DEVICES=6
docker run \
    --ipc=host \
    --gpus \"device=${CUDA_VISIBLE_DEVICES}\" \
    -v "$(pwd)":/src \
    --workdir /src \
    treetune_local \
    ./run.sh

# -v /export/fn05:/mnt \
