#!/bin/bash
#$ -N semantic_seg            # Job name
#$ -l h_rt=6:00:00            # Max wall-clock time  (317 frames × SAM2 ≈ 2-4 h on GPU)
#$ -l mem=32G                 # RAM per slot
#$ -l gpu=1                   # Request 1 GPU
#$ -l gpu_type=A100|V100|RTX  # Preferred GPU types (pipe = OR)
#$ -pe smp 4                  # 4 CPU threads for data loading
#$ -l tmpfs=20G               # Scratch space for HuggingFace weight cache
#$ -o logs/semantic_$JOB_ID.log
#$ -e logs/semantic_$JOB_ID.err
#$ -cwd                       # Run from submission directory
#$ -j n                       # Keep stdout/stderr separate

# ---------------------------------------------------------------------------
# UCL Myriad / Kathleen cluster – Grounded SAM2 semantic segmentation job
# ---------------------------------------------------------------------------
# Submit with:
#   mkdir -p logs
#   qsub ucl_gpu/run_semantic_job.sh
#
# Override label set or paths:
#   qsub -v LABELS="sofa,table,tv" ucl_gpu/run_semantic_job.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# ── 0. Logging helpers ──────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

log "Job $JOB_ID starting on host $(hostname)"
log "Working directory: $(pwd)"

# ── 1. Load cluster modules ─────────────────────────────────────────────────
module purge
module load gcc/10.2.0          || true
module load cuda/12.1.1         || module load cuda/11.8.0 || true
module load python3/3.11        || module load python3/3.10 || true

log "CUDA version: $(nvcc --version 2>/dev/null | tail -1 || echo 'nvcc not found')"
log "Python: $(python3 --version)"

# ── 2. Activate virtual environment ─────────────────────────────────────────
VENV_DIR="${HOME}/venvs/room3d"
if [[ -d "${VENV_DIR}" ]]; then
    source "${VENV_DIR}/bin/activate"
    log "Activated venv: ${VENV_DIR}"
else
    die "Virtual environment not found at ${VENV_DIR}.  Run setup first."
fi

# ── 3. Environment variables ─────────────────────────────────────────────────
export PYTORCH_ENABLE_MPS_FALLBACK=1        # no-op on CUDA, safe to set
export CUDA_VISIBLE_DEVICES=${SGE_GPU:-0}   # honour scheduler GPU assignment
export HF_HOME="${TMPDIR}/hf_cache"         # keep HF cache on fast scratch disk
export TORCH_HOME="${TMPDIR}/torch_cache"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

# Pre-downloaded weights (optional – script will auto-download otherwise)
WEIGHTS_DIR="$(pwd)/weights"
mkdir -p "${WEIGHTS_DIR}"

log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "HF_HOME=${HF_HOME}"

# ── 4. Verify GPU ────────────────────────────────────────────────────────────
python3 - <<'EOF'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU detected – check resource request")
gpu = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"[GPU] {gpu}  ({mem:.1f} GB VRAM)")
EOF

# ── 5. Configurable parameters ───────────────────────────────────────────────
FRAMES_DIR="${FRAMES_DIR:-data/mast3r_out/images}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/semantic}"
LABELS="${LABELS:-bed,desk,chair,laptop,shelf,door,window,fan,lamp,monitor}"
CONFIDENCE="${CONFIDENCE:-0.3}"
BATCH_SIZE="${BATCH_SIZE:-10}"

log "Frames dir  : ${FRAMES_DIR}"
log "Output dir  : ${OUTPUT_DIR}"
log "Labels      : ${LABELS}"
log "Confidence  : ${CONFIDENCE}"
log "Batch size  : ${BATCH_SIZE}"

# ── 6. Run segmentation ──────────────────────────────────────────────────────
log "Launching run_semantic.py …"

python3 scripts/run_semantic.py \
    --frames_dir  "${FRAMES_DIR}" \
    --output_dir  "${OUTPUT_DIR}" \
    --labels      "${LABELS}" \
    --device      cuda \
    --batch_size  "${BATCH_SIZE}" \
    --confidence  "${CONFIDENCE}"

EXIT_CODE=$?

# ── 7. Post-run reporting ────────────────────────────────────────────────────
log "Script exited with code ${EXIT_CODE}"

if [[ ${EXIT_CODE} -eq 0 ]]; then
    JSON_COUNT=$(find "${OUTPUT_DIR}" -maxdepth 1 -name "*.json" | wc -l)
    DEBUG_COUNT=$(find "${OUTPUT_DIR}/debug" -name "*_debug.png" 2>/dev/null | wc -l)
    log "Output summary:"
    log "  JSON files  : ${JSON_COUNT}"
    log "  Debug PNGs  : ${DEBUG_COUNT}"
    log "  Output dir  : ${OUTPUT_DIR}"
else
    die "run_semantic.py failed with exit code ${EXIT_CODE}"
fi

log "Job finished."
exit ${EXIT_CODE}