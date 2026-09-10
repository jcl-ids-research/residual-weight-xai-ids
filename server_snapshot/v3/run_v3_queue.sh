#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/ids_revision/v3_strengthened
SCRIPT_DIR="$ROOT/scripts"
RESULT_DIR="$ROOT/results"
LOG_DIR="$ROOT/logs"
STATUS_FILE="$ROOT/queue.status"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24
export CUDA_VISIBLE_DEVICES=0

log_status() {
    (
        flock -x 9
        printf '%s\n' "$1" >&9
    ) 9>>"$STATUS_FILE"
}

on_error() {
    code=$?
    log_status "FAILED exit=$code time=$(date -Iseconds)"
    exit "$code"
}
trap on_error ERR

printf 'STARTED time=%s pid=%s device=cuda concurrency=2 schema=v3\n' \
    "$(date -Iseconds)" "$$" >"$STATUS_FILE"

run_one() {
    dataset=$1
    seed=$2
    metric="$RESULT_DIR/$dataset/seed$seed/metrics.json"
    log="$LOG_DIR/${dataset}_seed${seed}.log"
    if [[ -s "$metric" ]]; then
        log_status "SKIP dataset=$dataset seed=$seed time=$(date -Iseconds)"
        return 0
    fi
    log_status "RUNNING dataset=$dataset seed=$seed device=cuda time=$(date -Iseconds)"
    (
        cd "$SCRIPT_DIR"
        nice -n 10 ionice -c 2 -n 7 /usr/bin/python3 run_v3_explainable.py \
            --dataset "$dataset" \
            --seed "$seed" \
            --out "$RESULT_DIR" \
            --device cuda \
            --n-jobs 24 \
            --shap-samples 1000 \
            >"$log" 2>&1
    )
    log_status "COMPLETE dataset=$dataset seed=$seed time=$(date -Iseconds)"
}

run_pair() {
    run_one "$1" "$2" &
    first_pid=$!
    run_one "$3" "$4" &
    second_pid=$!
    wait "$first_pid"
    wait "$second_pid"
}

for seed in 42 123 456 789 1001 2024 31415 65537 77777 99991; do
    run_pair unsw "$seed" nslkdd "$seed"
    log_status "PAIR_COMPLETE seed=$seed time=$(date -Iseconds)"
done

run_pair cicids2017 42 cicids2017 123
run_one cicids2017 456

cd "$SCRIPT_DIR"
nice -n 10 ionice -c 2 -n 7 /usr/bin/python3 aggregate_v3_results.py \
    --root "$RESULT_DIR" \
    --output "$ROOT/aggregate" \
    >"$LOG_DIR/aggregate.log" 2>&1
/usr/bin/python3 make_v3_figures.py \
    --summary "$ROOT/aggregate/summary.json" \
    --output "$ROOT/aggregate/figures" \
    >"$LOG_DIR/figures.log" 2>&1
log_status "ALL_COMPLETE time=$(date -Iseconds)"
