#!/bin/bash
set -e

echo "========================================="
echo "Starting SupportLLM End-to-End Pipeline"
echo "========================================="

# 1. Download data and model weights
echo -e "\n[Step 1/6] Downloading dataset and base model weights..."
python scripts/download_data.py

# 2. Run Hyperparameter Sweep (Optuna + MLflow)
echo -e "\n[Step 2/6] Running hyperparameter sweep..."
# We run sweep.py which runs 20 trials (using a subset of data to run quickly on CPU/MPS)
python scripts/sweep.py

# 3. Train Final Model with Best Config
echo -e "\n[Step 3/6] Training final model with best hyperparameters..."
python scripts/train.py --config configs/best.yaml

# 4. Evaluate Baseline vs Fine-tuned Model
echo -e "\n[Step 4/6] Evaluating model on test set..."
python scripts/eval.py

# 5. Start Serving API in background
echo -e "\n[Step 5/6] Starting FastAPI Serving server..."
python src/api.py &
API_PID=$!

# Wait for server to become healthy
echo "Waiting for API server to boot..."
MAX_ATTEMPTS=30
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "API server is healthy and ready!"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    sleep 2
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo "Error: API server failed to start within time limit."
    kill $API_PID || true
    exit 1
fi

# 6. Run Serving Benchmark
echo -e "\n[Step 6/6] Running serving throughput and latency benchmark..."
python scripts/benchmark_serving.py --num_requests 100 --concurrency 5

# Shutdown serving server
echo "Shutting down API server..."
kill $API_PID || true
wait $API_PID 2>/dev/null || true

echo "========================================="
echo "Pipeline completed successfully!"
echo "All results saved under results/"
echo "========================================="
