import time
import ollama

model = "llama3.2:1b"

# Warmup — loads model into memory, not timed
print("Warming up...")
ollama.chat(model=model, messages=[{"role": "user", "content": "hi"}])
print("Ready.\n")

# Now the real benchmark
prompt = "Explain the theory of relativity in simple terms."
messages = [{"role": "user", "content": prompt}]

first_token_time = None
start = time.perf_counter()
token_count = 0

stream = ollama.chat(model=model, messages=messages, stream=True)

for chunk in stream:
    content = chunk["message"]["content"]
    if content:
        if first_token_time is None:
            first_token_time = time.perf_counter()
            ttft = first_token_time - start
        token_count += 1
        print(content, end="", flush=True)

end = time.perf_counter()

total_time = end - start
decode_time = end - first_token_time
tps = token_count / decode_time

print(f"\n\n--- Benchmark ---")
print(f"TTFT:        {ttft*1000:.1f} ms")
print(f"Tokens:      {token_count}")
print(f"TPS:         {tps:.1f} tok/s")
print(f"Total time:  {total_time:.2f} s")