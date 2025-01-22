#!/bin/bash

echo 'Building Dockerfile with image name treetune_local'
docker build \
    --build-arg UID=$(id -u ${USER}) \
    --build-arg GID=1234 \
    --build-arg REQS="$(cat requirements.txt)" \
    -t treetune_local \
    --no-cache \
    .