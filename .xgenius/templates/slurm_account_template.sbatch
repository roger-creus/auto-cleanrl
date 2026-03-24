#!/bin/bash
#SBATCH --account={{ACCOUNT}}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={{NUM_CPUS}}
#SBATCH --gres=gpu:{{GPU_TYPE}}{{NUM_GPUS}}
#SBATCH --mem={{MEM}}
#SBATCH --time={{TIME}}
#SBATCH --output={{LOG_FILE}}
#SBATCH --job-name={{RUN_ID}}-{{EXPERIMENT_ID}}

mkdir -p {{OUTPUT_DIR_IN_CLUSTER}}
mkdir -p {{LOG_DIR}}

module load {{MODULES_TO_LOAD}}

echo "=== JOB DIAGNOSTICS ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_NODELIST=$SLURM_NODELIST"
echo "SLURM_MEM_PER_NODE=$SLURM_MEM_PER_NODE"
echo "HOSTNAME=$(hostname)"
echo "DATE=$(date -u)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed"
echo "========================"

# 1. Copy your container on the compute node
rsync -avz {{IMAGE_PATH}} $SLURM_TMPDIR

echo "Container rsync done at $(date -u)"

# 2. Executing your code with Apptainer
{{SINGULARITY_COMMAND}} exec --nv \
        -H {{CODE_DIR_IN_CLUSTER}} \
        --bind {{CODE_DIR_IN_CLUSTER}}:/src \
        --bind {{OUTPUT_DIR_IN_CLUSTER}}:{{OUTPUT_DIR_IN_CONTAINER}} \
        --workdir /src \
        $SLURM_TMPDIR/{{IMAGE_NAME}} \
        bash -c "export PYTHONDONTWRITEBYTECODE=1 && cd /src && {{COMMAND}}"
EXIT_CODE=$?
echo "=== JOB FINISHED ==="
echo "Exit code: $EXIT_CODE"
echo "Finished at: $(date -u)"
echo "===================="
