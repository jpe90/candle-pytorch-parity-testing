# XLM RoBERTa Fine-tuned Comparison

This directory compares inference implementations of [s-nlp/xlmr_formality_classifier](https://huggingface.co/s-nlp/xlmr_formality_classifier), a multilingual model that classifies text formality (e.g., "sup" vs "Hello, sir!").

## Implementations

- [`candle/`](./candle/): Rust implementation using Candle
- [`pytorch/`](./pytorch/): Python reference implementation using PyTorch

See each subfolder for specific implementation details and running instructions.

## Results

So far, Tokenization appears identical, but the logits coming out of the classifier head appear to be different.

```
Candle logits: [[2.0820313, -1.7548828], [0.7783203, -0.5629883], [1.2871094, -1.0039063], [2.1601563, -1.9277344]]
```

```
PyTorch logits: tensor([[ 2.6433, -2.3445],
        [ 1.0379, -0.9621],
        [ 1.4154, -1.2704],
        [ 3.4423, -3.1726]], grad_fn=<AddmmBackward0>)
```

The checksums match so the models appear to be the same. The configurations seem to match up; there isn't as much detailed info for Candle. More info below.

[troubleshooting.md](./troubleshooting.md)

