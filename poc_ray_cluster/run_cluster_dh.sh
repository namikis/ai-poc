#!/bin/bash

# Modified from vLLM examples/online_serving/run_cluster.sh
# Uses Docker Hub image by default instead of NGC image.

set -euo pipefail

HEAD_NODE_ADDRESS=${1:-}
NODE_TYPE=${2:-}
PATH_TO_HF_HOME=${3:-}
shift 3 || true

# Docker Hub image (pinned digest for reproducibility).
DOCKER_IMAGE_DEFAULT="vllm/vllm-openai:v0.15.1@sha256:06f9f0d5c7cb079504615c51dab70cd18abbf609d1358b940172181ac0a92efa"
DOCKER_IMAGE="${VLLM_IMAGE:-$DOCKER_IMAGE_DEFAULT}"

# By default, this script uses the host network to skip network setup in containers.
# If the container network setup is different, consult Ray documentation.
RAY_START_CMD="ray start --block"

# Additional arguments passed to Ray start command.
RAY_HEAD_ARGS=${RAY_HEAD_ARGS:-""}
RAY_WORKER_ARGS=${RAY_WORKER_ARGS:-""}

# Additional arguments passed to the model runner (vLLM serve by default).
# Keep this empty if you run head/worker setup first and start serving manually later.
LLM_COMMAND=${LLM_COMMAND:-""}

if [[ -z "$HEAD_NODE_ADDRESS" || -z "$NODE_TYPE" || -z "$PATH_TO_HF_HOME" ]]; then
    echo "Usage: $0 <HEAD_NODE_ADDRESS> <--head|--worker> <PATH_TO_HF_HOME> [ADDITIONAL_DOCKER_ARGS...]"
    echo "Example: $0 192.168.0.10 --head /path/to/huggingface_home"
    exit 1
fi

if [[ "$NODE_TYPE" == "--head" ]]; then
    RAY_CMD="$RAY_START_CMD --head --port=6379 $RAY_HEAD_ARGS"
elif [[ "$NODE_TYPE" == "--worker" ]]; then
    RAY_CMD="$RAY_START_CMD --address=${HEAD_NODE_ADDRESS}:6379 $RAY_WORKER_ARGS"
else
    echo "Error: 2nd argument must be --head or --worker"
    exit 1
fi

if [[ "$NODE_TYPE" == "--head" && -n "$LLM_COMMAND" ]]; then
    START_CMD="${RAY_CMD} & ${LLM_COMMAND}"
else
    START_CMD="${RAY_CMD}"
fi

# Ensure HF cache path exists and is writable on host.
mkdir -p "${PATH_TO_HF_HOME}"

# Pull image if not present locally.
if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    echo "Docker image not found locally. Pulling: $DOCKER_IMAGE"
    docker pull "$DOCKER_IMAGE"
fi

# If you run this in an environment that requires GPU runtime flags,
# append them as additional docker args, e.g.:
#   --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864

docker run \
    --entrypoint /bin/bash \
    --network host \
    --name "node" \
    --shm-size 10.24g \
    -v "${PATH_TO_HF_HOME}:/root/.cache/huggingface" \
    "$@" \
    "$DOCKER_IMAGE" \
    -c "
    ${START_CMD}
    "

# To start the OpenAI-compatible server immediately on the head node, set:
# export LLM_COMMAND='vllm serve /path/or/model --tensor-parallel-size 8 --pipeline-parallel-size 2'
# and replace the -c body above with:
#   ${RAY_CMD} &
#   ${LLM_COMMAND}
