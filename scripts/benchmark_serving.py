import os
import time
import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from concurrent.futures import ThreadPoolExecutor

def send_request(url, query, mode="classify"):
    """
    Sends a POST request to the server and measures latency.
    """
    data = json.dumps({"query": query, "max_tokens": 15}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            # Read all chunks (simulating streaming client consumption)
            body = response.read().decode("utf-8")
        latency = time.time() - start_time
        return True, latency, body
    except Exception as e:
        latency = time.time() - start_time
        return False, latency, str(e)

def run_benchmark(url, queries, concurrency=10):
    """
    Runs the benchmark for a list of queries with a given concurrency.
    """
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, url, q) for q in queries]
        for f in futures:
            success, latency, res = f.result()
            if success:
                results.append(latency)
                
    total_time = time.time() - start_time
    
    if not results:
        return 0.0, 0.0, 0.0, 0.0
        
    throughput = len(queries) / total_time
    p50 = np.percentile(results, 50)
    p95 = np.percentile(results, 95)
    avg_latency = np.mean(results)
    
    return throughput, p50, p95, avg_latency

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://localhost:8000")
    parser.add_argument("--num_requests", type=int, default=200) # scaled for quick local run
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    
    # Load dataset to get real user queries
    stats_path = "results/dataset_stats.json"
    if os.path.exists(stats_path):
        # We can construct some realistic queries or load from dataset
        try:
            from datasets import load_dataset
            dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
            queries = [x["instruction"] for x in dataset.select(range(args.num_requests))]
        except Exception:
            queries = [f"I want to track my order number {i}" for i in range(args.num_requests)]
    else:
        queries = [f"I want to track my order number {i}" for i in range(args.num_requests)]
        
    # Make sure we have enough queries
    if len(queries) < args.num_requests:
        queries = queries * (args.num_requests // len(queries) + 1)
    queries = queries[:args.num_requests]
    
    print(f"Starting benchmark: {args.num_requests} requests, concurrency={args.concurrency}")
    
    # Warmup
    print("Warming up server...")
    send_request(f"{args.host}/classify", "Warmup query")
    send_request(f"{args.host}/naive_classify", "Warmup query")
    time.sleep(1)
    
    # Test batching/vLLM endpoint
    print("\nBenchmarking Continuous Batching / vLLM endpoint...")
    vllm_url = f"{args.host}/classify"
    vllm_tp, vllm_p50, vllm_p95, vllm_avg = run_benchmark(vllm_url, queries, args.concurrency)
    print(f"Throughput: {vllm_tp:.2f} req/s | p50: {vllm_p50:.4f}s | p95: {vllm_p95:.4f}s")
    
    # Test Naive endpoint
    print("\nBenchmarking Naive Sequential endpoint...")
    naive_url = f"{args.host}/naive_classify"
    naive_tp, naive_p50, naive_p95, naive_avg = run_benchmark(naive_url, queries, args.concurrency)
    print(f"Throughput: {naive_tp:.2f} req/s | p50: {naive_p50:.4f}s | p95: {naive_p95:.4f}s")
    
    # Compute speedup multiplier
    speedup = vllm_tp / naive_tp if naive_tp > 0 else 0.0
    print(f"\nThroughput Speedup: {speedup:.2f}x")
    
    # Save serving_benchmark.csv
    csv_path = "results/serving_benchmark.csv"
    os.makedirs("results", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Continuous Batching (vLLM)", "Naive (HF)", "Speedup"])
        writer.writerow(["Throughput (req/s)", f"{vllm_tp:.4f}", f"{naive_tp:.4f}", f"{speedup:.2f}x"])
        writer.writerow(["p50 Latency (s)", f"{vllm_p50:.4f}", f"{naive_p50:.4f}", "N/A"])
        writer.writerow(["p95 Latency (s)", f"{vllm_p95:.4f}", f"{naive_p95:.4f}", "N/A"])
        writer.writerow(["Avg Latency (s)", f"{vllm_avg:.4f}", f"{naive_avg:.4f}", "N/A"])
    print(f"Saved benchmark results to {csv_path}")
    
    # Generate bar chart
    metrics = ['Throughput (req/s)', 'p50 Latency (s)', 'p95 Latency (s)']
    vllm_vals = [vllm_tp, vllm_p50, vllm_p95]
    naive_vals = [naive_tp, naive_p50, naive_p95]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, vllm_vals, width, label='Continuous Batching (vLLM)', color='#1f77b4')
    rects2 = ax.bar(x + width/2, naive_vals, width, label='Naive (HF)', color='#ff7f0e')
    
    ax.set_ylabel('Value')
    ax.set_title('Serving Performance Comparison: Continuous Batching vs Naive')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plot_path = "results/serving_comparison.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved comparison plot to {plot_path}")

if __name__ == "__main__":
    main()
