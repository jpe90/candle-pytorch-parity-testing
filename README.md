# Candle-PyTorch Parity Testing

This repository compares the implementations of various NLP models in [Candle](https://github.com/huggingface/candle) against PyTorch equivalents.

## Project Structure

The project is organized by model/task type:

- [`bert-embeddings/`](./bert-embeddings/): BERT embedding generation comparison
- [`xlm-roberta-finetuned/`](./xlm-roberta-finetuned/): compares inference on [s-nlp/xlmr_formality_classifier](https://huggingface.co/s-nlp/xlmr_formality_classifier) - XLM RoBERTa fine-tuned on the multilingual text classification task of determining formality of sentences

Each model directory contains:
- Subdirectories for each framework implementation (`candle/` and `pytorch/`)
- READMEs with specific implementation details
- Some results and discussion
