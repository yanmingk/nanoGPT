import torch
from fma import MultiheadAttention

def test_multihead_attention():
    batch_size = 2
    seq_length = 5
    embed_dim = 32
    num_heads = 4

    # Initialize random tensors for query, key, and value
    query = torch.rand(batch_size, seq_length, embed_dim)
    key = torch.rand(batch_size, seq_length, embed_dim)
    value = torch.rand(batch_size, seq_length, embed_dim)

    # Define key padding mask (e.g., zero out the attention for the last 2 elements in the sequences)
    key_padding_mask = torch.zeros(batch_size, seq_length).bool()
    key_padding_mask[:, -2:] = True

    # Initialize the MultiheadAttention module
    attention = MultiheadAttention(embed_dim, num_heads)

    # Without key padding mask
    output_without_mask, _ = attention(query, key, value, need_weights=True)
    print("Output without key padding mask:\n", output_without_mask)

    # With key padding mask
    output_with_mask, _ = attention(query, key, value, key_padding_mask=key_padding_mask, need_weights=True)
    print("Output with key padding mask:\n", output_with_mask)

if __name__ == "__main__":
    test_multihead_attention()
