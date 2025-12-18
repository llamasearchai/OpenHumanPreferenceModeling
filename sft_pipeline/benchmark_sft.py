import time

from transformers import AutoModelForCausalLM, AutoTokenizer


def benchmark_inference():
    # Load config
    # with open("configs/sft_config.yaml", "r") as f:
    #     config = yaml.safe_load(f)

    # Use tiny model for quick benchmark
    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "Test prompt for benchmarking latency."
    inputs = tokenizer(prompt, return_tensors="pt")

    # Warmup
    _ = model.generate(**inputs, max_new_tokens=10)

    # Measure
    latencies = []
    for _ in range(50):
        start = time.perf_counter()
        _ = model.generate(**inputs, max_new_tokens=20)
        latencies.append(time.perf_counter() - start)

    avg_lat = sum(latencies) / len(latencies)
    p99_lat = sorted(latencies)[int(0.99 * len(latencies))]

    print(f"Avg Latency: {avg_lat * 1000:.2f} ms")
    print(f"P99 Latency: {p99_lat * 1000:.2f} ms")


if __name__ == "__main__":
    benchmark_inference()
