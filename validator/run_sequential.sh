#!/bin/bash
# Sequential validator: GPT → Gemini → GLM, 10 workers each.
# Run from validator/ directory. Ensure DEEPSEEK_API_KEY is set.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

run_model() {
  model="$1"
  workers="$2"
  logprefix="/tmp/validate_${model}"

  echo ""
  echo "=========================================="
  echo " STARTING $model — $workers workers"
  echo "=========================================="

  for i in $(seq 1 "$workers"); do
    nohup python3 -u run_worker.py --model "$model" --shard "$i" --total-shards "$workers" \
      > "${logprefix}_${i}.log" 2>&1 &
    echo "  Worker $i/$workers: PID $!"
  done

  echo "  Waiting for all $model workers to finish..."
  wait

  echo ""
  echo "  All $model workers done. Merging shards..."
  python3 merge_shards.py --model "$model"
  echo "  $model complete."
  echo ""
}

echo "=== SEQUENTIAL VALIDATOR: GPT → Gemini → GLM ==="
echo "Workers: 10 each | Judge: deepseek-v4-pro | Timeout: 120s"

run_model gpt 10
run_model gemini 10
run_model glm 10

echo ""
echo "=== ALL MODELS COMPLETE ==="
echo "Results: $SCRIPT_DIR/results/{gpt,gemini,glm}/"
