#!/bin/bash
# =============================================================================
# RoboScene+ — MASt3R-SLAM Reconstruction Job (UCL bluestreak GPU)
# =============================================================================
#
# Run from the repo root on bluestreak (after `bash` + venv activated):
#
#   bash ucl_gpu/run_mast3r_job.sh
#
# The job runs in the background via nohup.  Watch progress with:
#   tail -f logs/mast3r_slam.log
#
# Stop the job:
#   kill $(cat logs/mast3r_slam.pid)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment — mirrors CLAUDE.md "Always Do These Two Things First" sequence
# ---------------------------------------------------------------------------

# 1. UCL Python 3.11 module (sets PATH so `python` resolves to 3.11)
if [[ -f /opt/Python/Python-3.11.5_Setup.csh ]]; then
    # The setup script is written for csh/tcsh; source the bash-equivalent
    # by prepending the interpreter bin directory directly.
    export PATH="/opt/Python/Python-3.11.5/bin:${PATH}"
fi

# 2. Activate the scratch venv (PyTorch 2.5.1+cu121)
# shellcheck disable=SC1091
source /scratch0/jrameshs/roboscene_env/bin/activate

# 3. MASt3R-SLAM internal imports
export PYTHONPATH=/scratch0/jrameshs/MASt3R-SLAM:${PYTHONPATH:-}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR=/scratch0/jrameshs/roboscene-plus
VIDEO_PATH=${REPO_DIR}/data/raw/room_video.mp4
OUTPUT_DIR=${REPO_DIR}/data/mast3r_out
MAST3R_DIR=/scratch0/jrameshs/MASt3R-SLAM
LOG_FILE=${REPO_DIR}/logs/mast3r_slam.log
PID_FILE=${REPO_DIR}/logs/mast3r_slam.pid

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
cd "${REPO_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

echo "========================================"
echo "  RoboScene+ — MASt3R-SLAM Job"
echo "  $(date)"
echo "========================================"
echo "  Repo     : ${REPO_DIR}"
echo "  Video    : ${VIDEO_PATH}"
echo "  Output   : ${OUTPUT_DIR}"
echo "  Log      : ${LOG_FILE}"
echo "  Python   : $(python --version 2>&1)"
echo "  Device   : $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'no GPU')"
echo "========================================"

if [[ ! -f "${VIDEO_PATH}" ]]; then
    echo "ERROR: video not found: ${VIDEO_PATH}"
    echo "Upload it first:"
    echo "  scp -J jrameshs@knuckles.cs.ucl.ac.uk \\"
    echo "      ~/3D-Spatial-Reconstruction/data/raw/room_video.mp4 \\"
    echo "      jrameshs@bluestreak.cs.ucl.ac.uk:${VIDEO_PATH}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Launch — nohup, stdout+stderr → log file
# ---------------------------------------------------------------------------
nohup python scripts/run_mast3r_slam.py \
    --video_path  data/raw/room_video.mp4 \
    --output_dir  data/mast3r_out/ \
    --mast3r_dir  /scratch0/jrameshs/MASt3R-SLAM/ \
    > "${LOG_FILE}" 2>&1 &

JOB_PID=$!
echo "${JOB_PID}" > "${PID_FILE}"

echo "  MASt3R-SLAM running, PID: ${JOB_PID}"
echo ""
echo "  Watch progress : tail -f ${LOG_FILE}"
echo "  Stop job       : kill \$(cat ${PID_FILE})"
echo "========================================"