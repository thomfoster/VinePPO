docker run \
  -v /export/fn05/thomf:/mnt \
  --ipc=host \
  --gpus all \
  treetune_local \
  df -h /mnt
