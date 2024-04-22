import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Non-JIT version
def block_matrix_multiply_algorithm_non_jit(X_blocks, Y, result, b, n, m, d, num_blocks):
    '''
        X_blocks: (b, num_blocks, m, 3*m)
        Y: (b, num_blocks, m, d)
        result: (b, n, d)
    '''
    for i in range(num_blocks):
        # Main diagonal blocks multiplication
        result[:, i*m:(i+1)*m, :] = torch.bmm(X_blocks[:, i, :, m:2*m], Y[:, i])

        # Superdiagonal blocks multiplication (except the first block)
        if i > 0:
            result[:, i*m:(i+1)*m, :] += torch.bmm(X_blocks[:, i, :, :m], Y[:, i-1])

        # Subdiagonal blocks multiplication (except the last block)
        if i < num_blocks - 1:
            result[:, i*m:(i+1)*m, :] += torch.bmm(X_blocks[:, i, :, 2*m:], Y[:, i+1])

# JIT version
@torch.jit.script
def block_matrix_multiply_algorithm_jit(X_blocks, Y, result, b: int, n: int, m: int, d: int, num_blocks: int):
    '''
        X_blocks: (b, num_blocks, m, 3*m)
        Y: (b, num_blocks, m, d)
        result: (b, n, d)
    '''
    for i in range(num_blocks):
        result[:, i*m:(i+1)*m, :] = torch.bmm(X_blocks[:, i, :, m:2*m], Y[:, i])

        if i > 0:
            result[:, i*m:(i+1)*m, :] += torch.bmm(X_blocks[:, i, :, :m], Y[:, i-1])

        if i < num_blocks - 1:
            result[:, i*m:(i+1)*m, :] += torch.bmm(X_blocks[:, i, :, 2*m:], Y[:, i+1])

b = 32 * 8
d = 96
m_values = [16, 32, 64]
n_values = [128, 256, 512, 1024, 2048, 4096, 8192]
results = {}
num_repeats = 5

for m in m_values:
    results[m] = {'n': [], 'time_jit': [], 'time_non_jit': [], 'memory_jit': [], 'memory_non_jit': []}
    for n in n_values:
        time_jit_list = []
        time_non_jit_list = []
        memory_jit_list = []
        memory_non_jit_list = []

        for _ in range(num_repeats):
            num_blocks = n // m
            X_blocks = torch.rand(b, num_blocks, m, 3*m, device=device)
            Y = torch.rand(b, n, d, device=device)
            Y = Y.view(b, num_blocks, m, d)

            if n != n_values[0]:
                result = torch.zeros(b, n, d, device=device)

                # JIT version
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                block_matrix_multiply_algorithm_jit(X_blocks, Y, result, b, n, m, d, num_blocks)
                end_event.record()
                torch.cuda.synchronize()

                time_jit_list.append(start_event.elapsed_time(end_event))
                memory_jit_list.append(torch.cuda.max_memory_allocated(device) / (1024**2))

                # Non-JIT version
                result.zero_()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                start_event.record()
                block_matrix_multiply_algorithm_non_jit(X_blocks, Y, result, b, n, m, d, num_blocks)
                end_event.record()
                torch.cuda.synchronize()

                time_non_jit_list.append(start_event.elapsed_time(end_event))
                memory_non_jit_list.append(torch.cuda.max_memory_allocated(device) / (1024**2))

        if n != n_values[0]:
            results[m]['n'].append(n)
            results[m]['time_jit'].append(np.mean(time_jit_list))
            results[m]['time_non_jit'].append(np.mean(time_non_jit_list))
            results[m]['memory_jit'].append(np.mean(memory_jit_list))
            results[m]['memory_non_jit'].append(np.mean(memory_non_jit_list))

plt.style.use('ggplot')
for m in m_values:
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Execution Time Comparison
    axs[0].plot(results[m]['n'], results[m]['time_jit'], 'o-', label='JIT', markersize=8)
    axs[0].plot(results[m]['n'], results[m]['time_non_jit'], 's-', label='Non-JIT', markersize=8)
    axs[0].set_title(f'Execution Time Comparison for m={m}', fontsize=14)
    axs[0].set_xlabel('n', fontsize=12)
    axs[0].set_ylabel('Time (ms)', fontsize=12)
    axs[0].legend()

    # Memory Usage Comparison
    axs[1].plot(results[m]['n'], results[m]['memory_jit'], 'o-', label='JIT', markersize=8)
    axs[1].plot(results[m]['n'], results[m]['memory_non_jit'], 's-', label='Non-JIT', markersize=8)
    axs[1].set_title(f'Memory Usage Comparison for m={m}', fontsize=14)
    axs[1].set_xlabel('n', fontsize=12)
    axs[1].set_ylabel('Memory (MB)', fontsize=12)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f'compare_cv_m_{m}.png')
