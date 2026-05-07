import time
from mlc_llm import MLCEngine

model = "HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC"
engine = MLCEngine(model)

# Warmup — compiles Metal kernels and loads weights into memory
print("Warming up...")
for _ in engine.chat.completions.create(
    messages=[{"role": "user", "content": "hi"}],
    model=model,
    stream=True,
):
    pass
print("Ready.\n")

# Real benchmark
prompt = "Explain the theory of relativity in simple terms."
messages = [{"role": "user", "content": prompt}]

first_token_time = None
start = time.perf_counter()
token_count = 0

for response in engine.chat.completions.create(
    messages=messages,
    model=model,
    stream=True,
):
    for choice in response.choices:
        if choice.delta.content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
                ttft = first_token_time - start
            token_count += 1
            print(choice.delta.content, end="", flush=True)

end = time.perf_counter()

total_time = end - start
decode_time = end - first_token_time
tps = token_count / decode_time

print(f"\n\n--- Benchmark ---")
print(f"TTFT:        {ttft*1000:.1f} ms")
print(f"Tokens:      {token_count}")
print(f"TPS:         {tps:.1f} tok/s")
print(f"Total time:  {total_time:.2f} s")

engine.terminate()