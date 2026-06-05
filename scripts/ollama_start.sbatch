#!/bin/bash
#SBATCH --job-name=OllamaServer
#SBATCH --output=/home/murban/logs/ollama.out
#SBATCH --error=/home/murban/logs/ollama.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch
#SBATCH --gpus=2
#SBATCH --mem=20GB
#SBATCH --signal=B:SIGTERM@30 # Tell Slurm to send SIGTERM to the batch script

echo "=== System Resources ==="
echo "Total RAM: $(free -h | grep Mem | awk '{print $2}')"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# 1. Define a cleanup function
cleanup() {
  echo "Caught cancellation signal. Stopping Ollama container..."
  docker stop ollama-server 2>/dev/null
  docker rm -f ollama-server 2>/dev/null
  echo "Cleanup complete. Exiting."
  exit 0
}

# 2. Trap the SIGTERM signal sent by scancel and trigger the cleanup function
trap cleanup SIGTERM SIGINT EXIT

# Initial clean up just in case
docker stop ollama-server 2>/dev/null && docker rm ollama-server 2>/dev/null

echo "Starting Ollama container..."

# 3. Add --rm and run the container in the background using '&'
docker run --rm \
  --gpus="\"device=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}\"" \
  --memory="256g" \
  --memory-swap="256g" \
  --shm-size="32g" \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ipc=host \
  -e OLLAMA_NUM_PARALLEL=1 \
  -e OLLAMA_MAX_LOADED_MODELS=1 \
  -e OLLAMA_KEEP_ALIVE=30m \
  -e OLLAMA_NUM_GPU=81 \
  -e OLLAMA_NUM_CTX=128000 \
  -e OLLAMA_FLASH_ATTENTION=1 \
  -e OLLAMA_DEBUG=1 \
  -v /mnt/labstore/SIGs/ML/ollama:/root/.ollama \
  -p 11434:11434 \
  --name ollama-server \
  ollama/ollama &

# 4. Capture the PID of the background docker process
DOCKER_PID=$!

# 5. Wait for the docker process. If scancel is run, 'wait' is interrupted and the trap fires.
wait $DOCKER_PID
