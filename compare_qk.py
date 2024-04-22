import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def block_matrix_multiply_non_jit(X_blocks, Y_blocks, result_blocks, num_blocks, m):
    '''
        X_blocks: (b, num_blocks, m, d)
        Y_blocks: (b, num_blocks, m, d)
        result_blocks: (b, num_blocks, m, 3*m)
    '''
    for i in range(num_blocks):
        if i > 0:
            result_blocks[:, i, :, :m] = torch.bmm(X_blocks[:, i], Y_blocks[:, i - 1].transpose(1, 2))
        result_blocks[:, i, :, m:2*m] = torch.bmm(X_blocks[:, i], Y_blocks[:, i].transpose(1, 2))
        if i < num_blocks - 1:
            result_blocks[:, i, :, 2*m:] = torch.bmm(X_blocks[:, i], Y_blocks[:, i + 1].transpose(1, 2))

@torch.jit.script
def block_matrix_multiply_jit(X_blocks, Y_blocks, result_blocks: torch.Tensor, num_blocks: int, m: int):
    '''
        X_blocks: (b, num_blocks, m, d)
        Y_blocks: (b, num_blocks, m, d)
        result_blocks: (b, num_blocks, m, 3*m)
    '''
    for i in range(num_blocks):
        if i > 0:
            result_blocks[:, i, :, :m] = torch.bmm(X_blocks[:, i], Y_blocks[:, i - 1].transpose(1, 2))
        result_blocks[:, i, :, m:2*m] = torch.bmm(X_blocks[:, i], Y_blocks[:, i].transpose(1, 2))
        if i < num_blocks - 1:
            result_blocks[:, i, :, 2*m:] = torch.bmm(X_blocks[:, i], Y_blocks[:, i + 1].transpose(1, 2))



@torch.jit.script
def qk_coarse(X_blocks, Y_blocks, result_blocks: torch.Tensor, num_blocks: int, ml: int, p: int):
    '''
        X_blocks: (b, num_blocks, ml, d)
        Y_blocks: (b, num_blocks, p, d)
        result_blocks: (b, num_blocks, ml, 4*p)
    '''
    for i in range(num_blocks):
        if i > 1:
            if i&1==0:
                result_blocks[:, i, :, :p] = torch.bmm(X_blocks[:, i], Y_blocks[:, i-2].transpose(1, 2))
            else:
                result_blocks[:, i, :, :p] = torch.bmm(X_blocks[:, i], Y_blocks[:, i-3].transpose(1, 2))
                result_blocks[:, i, :, p:2*p] = torch.bmm(X_blocks[:, i], Y_blocks[:, i-2].transpose(1, 2))
        if i < num_blocks - 2:
            if i&1==0:
                result_blocks[:, i, :, 2*p:3*p] = torch.bmm(X_blocks[:, i], Y_blocks[:, i+2].transpose(1, 2))
                result_blocks[:, i, :, 3*p:] = torch.bmm(X_blocks[:, i], Y_blocks[:, i+3].transpose(1, 2))
            else:
                result_blocks[:, i, :, 3*p:] = torch.bmm(X_blocks[:, i], Y_blocks[:, i+2].transpose(1, 2))


# Parameters setup
b = 32 * 8
d = 96
m_values = [16, 32, 64]
n_values = [128, 256, 512, 1024, 2048, 4096, 8192]
num_repeats = 5  # Number of repeats for each n
results = {}

# Measurement loop
for m in m_values:
    results[m] = {'n': [], 'time_jit': [], 'time_non_jit': [], 'memory_jit': [], 'memory_non_jit': []}
    for n in n_values:
        time_jit_list = []
        time_non_jit_list = []
        memory_jit_list = []
        memory_non_jit_list = []
        
        for _ in range(num_repeats):
            num_blocks = n // m
            X = torch.rand(b, n, d, device=device)
            Y = torch.rand(b, n, d, device=device)

            # JIT version
            torch.cuda.empty_cache()
            result_blocks_jit = torch.zeros(b, num_blocks, m, 3*m, device=device)
            X_blocks = X.view(b, num_blocks, m, d)
            Y_blocks = Y.view(b, num_blocks, m, d)

            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            block_matrix_multiply_jit(X_blocks, Y_blocks, result_blocks_jit, num_blocks, m)
            end_event.record()
            torch.cuda.synchronize()

            time_jit_list.append(start_event.elapsed_time(end_event))
            memory_jit_list.append(torch.cuda.max_memory_allocated(device) / (1024**2))

            # Non-JIT version
            torch.cuda.empty_cache()
            result_blocks_non_jit = torch.zeros(b, num_blocks, m, 3*m, device=device)

            torch.cuda.reset_peak_memory_stats(device)
            start_event.record()
            block_matrix_multiply_non_jit(X_blocks, Y_blocks, result_blocks_non_jit, num_blocks, m)
            end_event.record()
            torch.cuda.synchronize()

            time_non_jit_list.append(start_event.elapsed_time(end_event))
            memory_non_jit_list.append(torch.cuda.max_memory_allocated(device) / (1024**2))

        # Averaging results
        results[m]['n'].append(n)
        results[m]['time_jit'].append(np.mean(time_jit_list))
        results[m]['time_non_jit'].append(np.mean(time_non_jit_list))
        results[m]['memory_jit'].append(np.mean(memory_jit_list))
        results[m]['memory_non_jit'].append(np.mean(memory_non_jit_list))

# Plotting
# Improved plotting aesthetics
plt.style.use('ggplot')

for i, m in enumerate(m_values):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Execution Time Plot
    axs[0].plot(results[m]['n'][1:], results[m]['time_jit'][1:], 'o-', label='JIT', markersize=8)
    axs[0].plot(results[m]['n'][1:], results[m]['time_non_jit'][1:], 's-', label='Non-JIT', markersize=8)
    axs[0].set_title(f'Execution Time Comparison for m={m}', fontsize=14)
    axs[0].set_xlabel('n', fontsize=12)
    axs[0].set_ylabel('Time (ms)', fontsize=12)
    axs[0].legend()
    axs[0].grid(True)

    # Memory Usage Plot
    axs[1].plot(results[m]['n'][1:], results[m]['memory_jit'][1:], 'o-', label='JIT', markersize=8)
    axs[1].plot(results[m]['n'][1:], results[m]['memory_non_jit'][1:], 's-', label='Non-JIT', markersize=8)
    axs[1].set_title(f'Memory Usage Comparison for m={m}', fontsize=14)
    axs[1].set_xlabel('n', fontsize=12)
    axs[1].set_ylabel('Memory (MB)', fontsize=12)
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'compare_qk_m_{m}.png')
