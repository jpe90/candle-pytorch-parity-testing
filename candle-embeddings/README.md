# Candle Embeddings

Rust implementation using the Candle framework with `sentence-transformers/all-MiniLM-L6-v2` model.

## Quick Start

### Run

Default mode with sample sentences:
```bash
cargo run --release
```

With custom prompt:
```bash
cargo run --release -- --prompt "this is a test sentence"
```

### Options

Toggle normalization (default is false):
```bash
cargo run --release -- --normalize_embeddings true
cargo run --release -- --prompt "this is a test sentence" --normalize_embeddings true
```