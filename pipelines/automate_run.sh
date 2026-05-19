#!/bin/bash

# --- Stock AI Pipeline Automated Runner ---
# This script is intended to be run daily via CRON after market close.

PROJECT_DIR="/home/joka/personal/stockAiBackend"
LOG_DIR="$PROJECT_DIR/data/cron_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/run_$TIMESTAMP.log"

mkdir -p "$LOG_DIR"

echo "🚀 [$(date)] Starting Automated Pipeline..." | tee -a "$LOG_FILE"

# Move to project directory
cd "$PROJECT_DIR" || exit

# Run the pipeline using the local virtual environment
if [ -f "./venv/bin/python" ]; then
    ./venv/bin/python pipelines/run_all.py >> "$LOG_FILE" 2>&1
else
    echo "❌ [$(date)] Error: Virtual environment not found." | tee -a "$LOG_FILE"
    exit 1
fi

echo "✅ [$(date)] Pipeline finished. Log saved to $LOG_FILE" | tee -a "$LOG_FILE"
