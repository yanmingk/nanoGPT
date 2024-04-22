import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.jit.script
def block_matrix_multiply(X_blocks, Y_blocks, result_blocks: torch.Tensor, num_blocks: int, m: int):
    for i in range(num_blocks):
        # Handle the multiplication for block i with block i-1 in Y, if i > 0
        if i > 0:
            result_blocks[:, i, :, :m] = torch.bmm(X_blocks[:, i], Y_blocks[:, i - 1].transpose(1, 2))
        
        # Handle the multiplication for block i with block i in Y
        result_blocks[:, i, :, m:2*m] = torch.bmm(X_blocks[:, i], Y_blocks[:, i].transpose(1, 2))

        # Handle the multiplication for block i with block i+1 in Y, if i < num_blocks - 1
        if i < num_blocks - 1:
            result_blocks[:, i, :, 2*m:] = torch.bmm(X_blocks[:, i], Y_blocks[:, i + 1].transpose(1, 2))

# Now continue with your experiment using this corrected function

n_values = [2**i for i in range(5, 13)]  # From 32 to 4096
m = 32
results = []
num_iterations = 5

for n in n_values:
    b, d = 2, 4
    iteration_times = []
    iteration_peak_memories = []

    for _ in range(num_iterations):
        X = torch.rand(b, n, d).to(device)
        Y = torch.rand(b, n, d).to(device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        num_blocks = n // m
        result_blocks = torch.zeros(b, num_blocks, m, 3*m, device=device)
        X_blocks = X.view(b, num_blocks, m, d)
        Y_blocks = Y.view(b, num_blocks, m, d)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        block_matrix_multiply(X_blocks, Y_blocks, result_blocks, num_blocks, m)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time = start_event.elapsed_time(end_event)
        iteration_times.append(elapsed_time / 1000)

        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
        iteration_peak_memories.append(peak_memory)

        print(result_blocks.shape)

    avg_time = np.mean(iteration_times)
    avg_peak_memory = np.mean(iteration_peak_memories)
    results.append((n, m, avg_time, avg_peak_memory))

for n, m, avg_time, avg_peak_memory in results:
    print(f"n: {n}, m: {m} -> Average execution time: {avg_time} seconds, Average peak memory footprint: {avg_peak_memory} MB")
