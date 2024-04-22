import torch

# Sample input tensors of shape [b, n, d]
b, n, d, m = 2, 6, 4, 2  # Example dimensions; ensure n % m == 0
X = torch.randn(b, n, d)
Y = torch.randn(b, d, n)  # Adjusted for multiplication

# Initialize the result tensor, now with shape [b, n, 3m]
result = torch.zeros(b, n, 3 * m)

# Define block size and number of blocks
num_blocks = n // m

# Process block tri-diagonals
for i in range(num_blocks):
    # Main diagonal block multiplication and storage
    main_block = torch.bmm(X[:, i*m:(i+1)*m, :], Y[:, :, i*m:(i+1)*m])
    result[:, i*m:(i+1)*m, m:2*m] = main_block

    if i > 0:
        # Upper diagonal block (left side multiplication and storage)
        upper_block = torch.bmm(X[:, (i-1)*m:i*m, :], Y[:, :, i*m:(i+1)*m])
        result[:, i*m:(i+1)*m, :m] = upper_block

    if i < num_blocks - 1:
        # Lower diagonal block (right side multiplication and storage)
        lower_block = torch.bmm(X[:, (i+1)*m:(i+2)*m, :], Y[:, :, i*m:(i+1)*m])
        result[:, i*m:(i+1)*m, 2*m:3*m] = lower_block


print(result.shape)

import torch


# Initialize the result tensor, now with shape [b, n, 3m]
result2 = torch.zeros(b, n, 3 * m)

# Define block size and number of blocks
num_blocks = n // m

# Diagonal blocks for X and Y
diag_X = X.view(b, num_blocks, m, d)
diag_Y = Y.transpose(1, 2).reshape(b, num_blocks, m, d)

# Upper diagonal blocks for X, shifting one block up
upper_X = X[:, :-m].reshape(b, num_blocks - 1, m, d)
upper_Y = Y.transpose(1, 2)[:, m:].reshape(b, num_blocks - 1, m, d)

# Lower diagonal blocks for X, shifting one block down
lower_X = X[:, m:].reshape(b, num_blocks - 1, m, d)
lower_Y = Y.transpose(1, 2)[:, :-m].reshape(b, num_blocks - 1, m, d)

# Perform batch matrix multiplications in parallel
diag_mul = torch.bmm(diag_X.reshape(-1, m, d), diag_Y.reshape(-1, d, m)).reshape(b, num_blocks, m, m)
upper_mul = torch.bmm(upper_X.reshape(-1, m, d), upper_Y.reshape(-1, d, m)).reshape(b, num_blocks - 1, m, m)
lower_mul = torch.bmm(lower_X.reshape(-1, m, d), lower_Y.reshape(-1, d, m)).reshape(b, num_blocks - 1, m, m)

# Properly assign the multiplied blocks into the result tensor
# For upper blocks
result2[:, m:n - m, :m] = upper_mul.reshape(b, (num_blocks - 1) * m, m)

# For diagonal blocks (reshape might be unnecessary depending on how you handle it, just ensuring clarity here)
result2[:, :n, m:2 * m] = diag_mul.reshape(b, num_blocks * m, m)

# For lower blocks
result2[:, m:n - m, 2 * m:3 * m] = lower_mul.reshape(b, (num_blocks - 1) * m, m)
