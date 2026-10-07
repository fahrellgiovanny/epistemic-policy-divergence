#!/bin/bash
# Launch 12 validator workers (4 per model) in parallel.
# Run from validator/ directory. Ensure DEEPSEEK_API_KEY is set.
# 12 workers x ~25 req/min = ~300 req/min (DeepSeek limit: 500)

echo "Launching 12 workers..."
echo ""

# GPT: 4 workers (72 files / 4 = 18 each)
for i in 1 2 3 4; do
  nohup python3 -u run_worker.py --model gpt --shard $i --total-shards 4 > /tmp/validate_gpt_$i.log 2>&1 &
  echo "GPT worker $i: PID $!"
done
echo ""

# Gemini: 4 workers (72 files / 4 = 18 each)
for i in 1 2 3 4; do
  nohup python3 -u run_worker.py --model gemini --shard $i --total-shards 4 > /tmp/validate_gemini_$i.log 2>&1 &
  echo "Gemini worker $i: PID $!"
done
echo ""

# GLM: 4 workers (72 files / 4 = 18 each)
for i in 1 2 3 4; do
  nohup python3 -u run_worker.py --model glm --shard $i --total-shards 4 > /tmp/validate_glm_$i.log 2>&1 &
  echo "GLM worker $i: PID $!"
done

echo ""
echo "All 12 workers launched."
echo "Check: tail -f /tmp/validate_gpt_1.log"
echo "Kill:  pkill -f run_worker.py"
