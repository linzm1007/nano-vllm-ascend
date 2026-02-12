# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Nano-vLLM project


import os
import time
import numpy as np
import argparse
from random import randint, seed
from tqdm.auto import tqdm
from nanovllm import LLM, SamplingParams

# --- Constants ---
MODEL_PATH = os.path.expanduser("/data_mount/models/Qwen3-0.6B/")
MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 1024

# --- Seed for reproducibility ---
seed(0)
np.random.seed(0)


class RequestMetrics:
    """Stores metrics for a single request."""

    def __init__(self, request_id, input_len, max_output_len):
        self.request_id = request_id
        self.input_len = input_len
        self.max_output_len = max_output_len
        self.submission_time = -1
        self.first_token_time = -1
        self.completion_time = -1
        self.output_len = -1

    def record_submission(self):
        self.submission_time = time.perf_counter()

    def record_first_token(self):
        if self.first_token_time == -1:
            self.first_token_time = time.perf_counter()

    def record_completion(self, output_ids):
        self.completion_time = time.perf_counter()
        self.output_len = len(output_ids)

    @property
    def ttft(self):
        return self.first_token_time - self.submission_time

    @property
    def tpot(self):
        if self.output_len > 1:
            return (self.completion_time - self.first_token_time) / (self.output_len - 1)
        return float('nan')

    @property
    def latency(self):
        return self.completion_time - self.submission_time


def main():
    """Main function to run the serving benchmark."""
    parser = argparse.ArgumentParser(description="Serving benchmark for nano-vllm.")
    parser.add_argument("--num-requests", type=int, default=256, help="Number of requests to process.")
    parser.add_argument("--request-rate", type=int, default=8, help="Request rate (requests per second).")
    args = parser.parse_args()

    num_requests = args.num_requests
    request_rate = args.request_rate

    print(f"\n--- Running benchmark with --num-requests {num_requests} --request-rate {request_rate} ---")
    llm = LLM(MODEL_PATH, enforce_eager=False, max_model_len=4096)
    engine = llm

    # --- Generate random prompts ---
    prompts = [[randint(0, 10000) for _ in range(randint(100, MAX_INPUT_LEN))] for _ in range(num_requests)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, MAX_OUTPUT_LEN)) for _
                       in range(num_requests)]

    # --- Generate request arrival times ---
    request_intervals = np.random.poisson(1.0 / request_rate, num_requests)
    arrival_times = np.cumsum(request_intervals)

    # --- Benchmark loop ---
    metrics = {}
    requests_sent = 0
    start_time = time.perf_counter()
    completed_latencies = []

    with tqdm(total=num_requests, desc="Processing Requests") as pbar:
        while requests_sent < num_requests or not engine.is_finished():
            # --- Send new requests ---
            current_time = time.perf_counter()
            while requests_sent < num_requests and current_time - start_time >= arrival_times[requests_sent]:
                prompt = prompts[requests_sent]
                sp = sampling_params[requests_sent]

                engine.add_request(prompt, sp)

                new_seq = engine.scheduler.waiting[-1]
                seq_id = new_seq.seq_id
                req_metrics = RequestMetrics(seq_id, len(prompt), sp.max_tokens)
                req_metrics.record_submission()
                metrics[seq_id] = req_metrics

                requests_sent += 1

            # --- Engine step ---
            if engine.scheduler.waiting or engine.scheduler.running:
                finished_outputs, _ = engine.step()

                # Record first token time for all processed sequences
                all_processed_seqs = list(engine.scheduler.running)
                for seq in all_processed_seqs:
                    if seq.seq_id in metrics:
                        metrics[seq.seq_id].record_first_token()

                for seq_id, output_ids in finished_outputs:
                    if seq_id in metrics:
                        metrics[seq_id].record_first_token()  # Ensure first token time is recorded
                        metrics[seq_id].record_completion(output_ids)

                        completed_latencies.append(metrics[seq_id].latency)
                        avg_latency = np.mean(completed_latencies)
                        pbar.set_postfix({"Avg Latency": f"{avg_latency:.2f}s"})
                        pbar.update(1)
            else:
                # If no requests are running or waiting, sleep briefly
                time.sleep(0.01)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # --- Calculate and print metrics ---
    total_input_tokens = sum(m.input_len for m in metrics.values())
    total_output_tokens = sum(m.output_len for m in metrics.values() if m.output_len != -1)

    avg_ttft = np.mean([m.ttft for m in metrics.values() if m.first_token_time != -1])
    avg_tpot = np.mean([m.tpot for m in metrics.values() if not np.isnan(m.tpot)])
    avg_latency = np.mean([m.latency for m in metrics.values() if m.completion_time != -1])
    throughput = total_output_tokens / total_time

    print("--- Benchmark Results ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests sent: {requests_sent}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Average TTFT: {avg_ttft * 1000:.2f} ms")
    print(f"Average TPOT: {avg_tpot * 1000:.2f} ms/token")
    print(f"Average latency: {avg_latency:.2f} s")
    print("-------------------------\n")


if __name__ == "__main__":
    main()
