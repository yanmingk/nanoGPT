import torch

def block_tri_diagonal_mm(X, Y, m):
    # Get dimensions
    b, n, d = X.size()

    # Ensure n is divisible by m
    assert n % m == 0

    # Calculate the number of blocks
    num_blocks = n // m

    # Reshape X and Y to have block dimensions
    X_blocks = X.view(b, num_blocks, m, d)
    Y_blocks = Y.view(b, num_blocks, m, d)

    # Initialize result tensor
    result = torch.zeros(b, num_blocks, m, 3 * m, dtype=X.dtype, device=X.device)

    # Perform block tri-diagonal matrix multiplication
    for i in range(num_blocks):
        # Diagonal blocks
        result[:, i, :, :m] += torch.bmm(X_blocks[:, i, :, :], Y_blocks[:, i, :, :].transpose(1, 2))

        # Off-diagonal blocks
        if i > 0:
            result[:, i, :, m:2*m] += torch.bmm(X_blocks[:, i, :, :], Y_blocks[:, i - 1, :, :].transpose(1, 2))
        if i < num_blocks - 1:
            result[:, i, :, 2*m:] += torch.bmm(X_blocks[:, i, :, :], Y_blocks[:, i + 1, :, :].transpose(1, 2))

    return result

# Example usage
b, n, d = 2, 6, 4
m = 2
X = torch.rand(b, n, d).to('cuda')
Y = torch.rand(b, n, d).to('cuda')

result = block_tri_diagonal_mm(X, Y, m)
print(result.shape)  # Output: torch.Size([2, 3, 2, 6])
