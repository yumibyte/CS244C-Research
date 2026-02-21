#!/bin/bash
NUM_GPUS=${1}

# Capture directory where script was launched
BASE_DIR="$(pwd)"

NCCL_HOME="/home/users/raigosa/micromamba/envs/nccl-env"

srun --export=ALL --partition=gpu --gres="gpu:${NUM_GPUS}" --nodes=1 bash -c '

export NUM_GPUS='"${NUM_GPUS}"'
export NCCL_HOME='"${NCCL_HOME}"'
export BASE_DIR='"${BASE_DIR}"'

export PATH="$HOME/micromamba/bin:$PATH"
export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
__mamba_setup="$($MAMBA_EXE shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
  eval "$__mamba_setup"
else
  alias micromamba="$MAMBA_EXE"
fi
unset __mamba_setup

module load cuda/12.9.0

micromamba run -n nccl-env bash -c "
export CUDA_HOME=\$(dirname \$(dirname \$(which nvcc)))

if [[ -z \"\$CUDA_HOME\" || ! -d \"\$CUDA_HOME\" ]]; then
  echo Error: CUDA_HOME invalid
  exit 1
fi

if [[ -z \"\$NCCL_HOME\" || ! -d \"\$NCCL_HOME\" ]]; then
  echo Error: NCCL_HOME invalid
  exit 1
fi

export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=\$NCCL_HOME/lib:\$LD_LIBRARY_PATH

cd ../../nccl-tests
make MPI=0 CUDA_HOME=\"\$CUDA_HOME\" NCCL_HOME=\"\$NCCL_HOME\"

cd build

GPU_NAME=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
GPU_TAG=\$(echo \"\$GPU_NAME\" | tr '\''[:upper:]'\'' '\''[:lower:]'\'' | tr -d '\'' '\'')

RESULT_DIR=\${BASE_DIR}/../\${GPU_TAG}_\${NUM_GPUS}gpu_results
mkdir -p \"\$RESULT_DIR\"

TIMESTAMP=\$(date +%Y-%m-%d_%H-%M-%S)

./all_reduce_perf -b 8 -e 128M -f 2 -g \$NUM_GPUS | tee \"\$RESULT_DIR/\${TIMESTAMP}.txt\"

echo \"Results saved to \$RESULT_DIR/\${TIMESTAMP}.txt\"
"
'