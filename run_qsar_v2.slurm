#!/bin/bash
#SBATCH --job-name=qsar_v2
#SBATCH --output=qsar_v2_%j.out
#SBATCH --error=qsar_v2_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --account=p307

set -e
echo "Job $SLURM_JOB_ID on $(hostname) | CPUs=$SLURM_CPUS_PER_TASK | start $(date)"

source /eb/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate qsar_env

export OMP_NUM_THREADS=1          # مهم: نخلي التوازي عبر CV folds مو داخل كل موديل
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
python -c "import rdkit,sklearn,xgboost; print('rdkit',rdkit.__version__,'sklearn',sklearn.__version__,'xgb',xgboost.__version__)"
python -c "import mordred; print('mordred OK')" || echo "تنبيه: mordred غير مثبت — سيرجع لـ RDKit descriptors"

echo "------ RUN ------"
time python qsar_builder_v2.py

echo "Done $(date) | outputs in $SLURM_SUBMIT_DIR/qsar_output_v2/"
