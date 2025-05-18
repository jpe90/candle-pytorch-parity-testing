# BERT Embeddings Comparison

This directory contains parity testing for BERT embedding implementations in Candle and PyTorch.


The goal of this project is to ensure that Candle and PyTorch implementations produce equivalent:
- Output values (embeddings, inference results, etc.)
- Mathematical operations (normalization, similarity calculations)
- Practical behavior across common use cases

This helps validate that Candle can be a viable replacement for PyTorch in production settings where Rust is preferred.

For example, comparing the implementation of L2 normalization:

```Rust
pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
```

vs

```Python
def normalize_l2(embeddings):
    return F.normalize(embeddings, p=2, dim=1)
```

## Results

Both implementaitons produced identical embeddings, both raw and normalized. They also had identical cosine similarity scores.

### Candle

```bash
➜  candle-embeddings git:(master) ✗ cargo run --bin embeddings -- --prompt "this is a test"
   Compiling embeddings v0.1.0 (/Users/jon/development/ai/parity/candle-embeddings)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.83s
     Running `target/debug/embeddings --prompt 'this is a test'`
prompt: this is a test
[[[-0.0058, -0.1078, -0.0312, ..., -0.1729, -0.0137, -0.2952],
  [ 0.6984,  0.0371, -0.1464, ..., -0.2448,  1.0019,  0.4158],
  [-0.1385,  0.2582,  0.1129, ...,  0.1124,  1.0157, -0.0616],
  [-0.6541,  0.0111,  0.0174, ..., -0.3531,  0.1437,  0.4319],
  [ 0.1310,  0.1267, -0.2294, ...,  0.5801,  0.3392, -0.7176],
  [ 0.9691,  0.1266, -0.4043, ..., -0.4712,  0.5899, -1.2384]]]
Tensor[[1, 6, 384], f32]
➜  candle-embeddings git:(master) ✗ cargo run --bin embeddings -- --prompt "this is a test" --normalize-embeddings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.12s
     Running `target/debug/embeddings --prompt 'this is a test' --normalize-embeddings`
prompt: this is a test
[[[-0.0042, -0.3222, -0.0621, ..., -0.1951, -0.0086, -0.1867],
  [ 0.5079,  0.1110, -0.2919, ..., -0.2762,  0.6312,  0.2630],
  [-0.1007,  0.7721,  0.2251, ...,  0.1268,  0.6399, -0.0389],
  [-0.4757,  0.0331,  0.0348, ..., -0.3984,  0.0905,  0.2732],
  [ 0.0953,  0.3787, -0.4575, ...,  0.6545,  0.2137, -0.4540],
  [ 0.7047,  0.3784, -0.8061, ..., -0.5316,  0.3717, -0.7834]]]
Tensor[[1, 6, 384], f32]
➜  candle-embeddings git:(master) ✗ cargo run --bin embeddings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.11s
     Running `target/debug/embeddings`
running inference on batch [8, 8]
generated embeddings [8, 8, 384]
pooled embeddings [8, 384]
score: 0.87 'The new movie is awesome' 'The new movie is so great'
score: 0.65 'The cat sits outside' 'The cat plays in the garden'
score: 0.52 'I love pasta' 'Do you like pizza?'
score: 0.28 'I love pasta' 'The new movie is awesome'
score: 0.22 'The new movie is awesome' 'Do you like pizza?'
```

### Python
```bash
(.venv) ➜  pytorch-embeddings git:(master) ✗ python main.py --prompt "this is a test"
prompt: this is a test
Raw output shape: torch.Size([1, 6, 384])
tensor([[[-0.0058, -0.1078, -0.0312,  ..., -0.1729, -0.0137, -0.2952],
         [ 0.6984,  0.0371, -0.1464,  ..., -0.2448,  1.0019,  0.4158],
         [-0.1385,  0.2582,  0.1129,  ...,  0.1124,  1.0157, -0.0616],
         [-0.6541,  0.0111,  0.0174,  ..., -0.3531,  0.1437,  0.4319],
         [ 0.1310,  0.1267, -0.2294,  ...,  0.5801,  0.3392, -0.7176],
         [ 0.9691,  0.1266, -0.4043,  ..., -0.4712,  0.5899, -1.2384]]])
(.venv) ➜  pytorch-embeddings git:(master) ✗ python main.py --prompt "this is a test" --normalize_embeddings=True
prompt: this is a test
Raw output shape: torch.Size([1, 6, 384])
tensor([[[-0.0042, -0.3222, -0.0621,  ..., -0.1951, -0.0086, -0.1867],
         [ 0.5079,  0.1110, -0.2919,  ..., -0.2762,  0.6312,  0.2630],
         [-0.1007,  0.7721,  0.2251,  ...,  0.1268,  0.6399, -0.0389],
         [-0.4757,  0.0331,  0.0348,  ..., -0.3984,  0.0905,  0.2732],
         [ 0.0953,  0.3787, -0.4575,  ...,  0.6545,  0.2137, -0.4540],
         [ 0.7047,  0.3784, -0.8061,  ..., -0.5316,  0.3717, -0.7834]]])
(.venv) ➜  pytorch-embeddings git:(master) ✗ python main.py
running inference on batch torch.Size([8, 8])
generated embeddings torch.Size([8, 8, 384])
pooled embeddings torch.Size([8, 384])
score: 0.87 'The new movie is awesome' 'The new movie is so great'
score: 0.65 'The cat sits outside' 'The cat plays in the garden'
score: 0.52 'I love pasta' 'Do you like pizza?'
score: 0.28 'I love pasta' 'The new movie is awesome'
score: 0.22 'The new movie is awesome' 'Do you like pizza?'
```

## Discussion

I wanted to try this on a simpler and older model to make sure the approach is valid before exploring similar efforts to validate the behavior of newer, more complex BERT-like models.
