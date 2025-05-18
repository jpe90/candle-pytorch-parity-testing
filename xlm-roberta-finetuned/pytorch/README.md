# PyTorch XLM RoBERTa Fine-tuned Implementation

This program compares inference on [s-nlp/xlmr_formality_classifier](https://huggingface.co/s-nlp/xlmr_formality_classifier) - XLM RoBERTa fine-tuned on the multilingual text classification task of determining formality of sentences (formality as in, "sup" vs. "Hello, sir!").

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python main.py
```