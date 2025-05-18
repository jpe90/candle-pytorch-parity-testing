# PyTorch BERT Embeddings

Python reference implementation using PyTorch and Transformers with `sentence-transformers/all-MiniLM-L6-v2` model.

## Overview

This implementation serves as the reference for the Candle BERT implementation, demonstrating:
- Hugging Face Transformers API usage for BERT
- PyTorch's built-in normalization functions
- Standard cosine similarity calculations

## Quick Start

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

Default mode with sample sentences:
```bash
python main.py
```

With custom prompt:
```bash
python main.py --prompt "this is a test sentence"
```

### Options

Toggle normalization (default is false):
```bash
python main.py --normalize_embeddings True
python main.py --prompt "this is a test sentence" --normalize_embeddings True
```
