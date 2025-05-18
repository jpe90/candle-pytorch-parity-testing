# XLM RoBERTa Fine-tuned Comparison

This directory compares inference implementations of [s-nlp/xlmr_formality_classifier](https://huggingface.co/s-nlp/xlmr_formality_classifier), a multilingual model that classifies text formality (e.g., "sup" vs "Hello, sir!").

## Implementations

- [`candle/`](./candle/): Rust implementation using Candle
- [`pytorch/`](./pytorch/): Python reference implementation using PyTorch

See each subfolder for specific implementation details and running instructions.

## Results

So far, Tokenization appears identical, but the Formality Scores are slightly different. Will need to investigate.

### Tokenization

#### PyTorch

```Python
(.venv) ➜  pytorch git:(master) ✗ python main.py
========== TOKENIZATION INFO ==========

Text 1: "I like you. I love you"
Tokens: ['▁I', '▁like', '▁you', '.', '▁I', '▁love', '▁you']
Token IDs: [0, 87, 1884, 398, 5, 87, 5161, 398, 2]
Token ID to Token Mapping:
  87 → ▁I
  1884 → ▁like
  398 → ▁you
  5 → .
  87 → ▁I
  5161 → ▁love
  398 → ▁you
----------------------------------------

Text 2: "Hey, what's up?"
Tokens: ['▁Hey', ',', '▁what', "'", 's', '▁up', '?']
Token IDs: [0, 28240, 4, 2367, 25, 7, 1257, 32, 2]
Token ID to Token Mapping:
  28240 → ▁Hey
  4 → ,
  2367 → ▁what
  25 → '
  7 → s
  1257 → ▁up
  32 → ?
----------------------------------------

Text 3: "Siema, co porabiasz?"
Tokens: ['▁Sie', 'ma', ',', '▁co', '▁po', 'rabia', 'sz', '?']
Token IDs: [0, 727, 192, 4, 552, 160, 74883, 1330, 32, 2]
Token ID to Token Mapping:
  727 → ▁Sie
  192 → ma
  4 → ,
  552 → ▁co
  160 → ▁po
  74883 → rabia
  1330 → sz
  32 → ?
----------------------------------------

Text 4: "I feel deep regret and sadness about the situation in international politics."
Tokens: ['▁I', '▁feel', '▁deep', '▁regret', '▁and', '▁sad', 'ness', '▁about', '▁the', '▁situation', '▁in', '▁international', '▁politic', 's', '.']
Token IDs: [0, 87, 12319, 53894, 84377, 136, 17110, 7432, 1672, 70, 16648, 23, 21640, 44951, 7, 5, 2]
Token ID to Token Mapping:
  87 → ▁I
  12319 → ▁feel
  53894 → ▁deep
  84377 → ▁regret
  136 → ▁and
  17110 → ▁sad
  7432 → ness
  1672 → ▁about
  70 → ▁the
  16648 → ▁situation
  23 → ▁in
  21640 → ▁international
  44951 → ▁politic
  7 → s
  5 → .
----------------------------------------
```

#### Candle

```Rust
========== TOKENIZATION INFO ==========

Text 1: "I like you. I love you"
Tokens: ["<s>", "▁I", "▁like", "▁you", ".", "▁I", "▁love", "▁you", "</s>"]
Token IDs: [0, 87, 1884, 398, 5, 87, 5161, 398, 2]
Token ID to Token Mapping:
  87 → ▁I
  1884 → ▁like
  398 → ▁you
  5 → .
  87 → ▁I
  5161 → ▁love
  398 → ▁you
----------------------------------------

Text 2: "Hey, what's up?"
Tokens: ["<s>", "▁Hey", ",", "▁what", "'", "s", "▁up", "?", "</s>"]
Token IDs: [0, 28240, 4, 2367, 25, 7, 1257, 32, 2]
Token ID to Token Mapping:
  28240 → ▁Hey
  4 → ,
  2367 → ▁what
  25 → '
  7 → s
  1257 → ▁up
  32 → ?
----------------------------------------

Text 3: "Siema, co porabiasz?"
Tokens: ["<s>", "▁Sie", "ma", ",", "▁co", "▁po", "rabia", "sz", "?", "</s>"]
Token IDs: [0, 727, 192, 4, 552, 160, 74883, 1330, 32, 2]
Token ID to Token Mapping:
  727 → ▁Sie
  192 → ma
  4 → ,
  552 → ▁co
  160 → ▁po
  74883 → rabia
  1330 → sz
  32 → ?
----------------------------------------

Text 4: "I feel deep regret and sadness about the situation in international politics."
Tokens: ["<s>", "▁I", "▁feel", "▁deep", "▁regret", "▁and", "▁sad", "ness", "▁about", "▁the", "▁situation", "▁in", "▁international", "▁politic", "s", ".", "</s>"]
Token IDs: [0, 87, 12319, 53894, 84377, 136, 17110, 7432, 1672, 70, 16648, 23, 21640, 44951, 7, 5, 2]
Token ID to Token Mapping:
  87 → ▁I
  12319 → ▁feel
  53894 → ▁deep
  84377 → ▁regret
  136 → ▁and
  17110 → ▁sad
  7432 → ness
  1672 → ▁about
  70 → ▁the
  16648 → ▁situation
  23 → ▁in
  21640 → ▁international
  44951 → ▁politic
  7 → s
  5 → .
----------------------------------------
```
### Formality Scores

#### PyTorch

```Python
Formality Scores:
Text 1: "I like you. I love you"
  formal: 0.9932
  informal: 0.0068

Text 2: "Hey, what's up?"
  formal: 0.8808
  informal: 0.1192

Text 3: "Siema, co porabiasz?"
  formal: 0.9362
  informal: 0.0638

Text 4: "I feel deep regret and sadness about the situation in international politics."
  formal: 0.9987
  informal: 0.0013
  ```
  
#### Candle
  
```Rust
  Predictions: [0, 0, 0, 0]

Formality Scores:
--------------------------------------------------------------------------------
Text 1: "I like you. I love you"
  formal: 0.9789
  informal: 0.0211

Text 2: "Hey, what's up?"
  formal: 0.7927
  informal: 0.2073

Text 3: "Siema, co porabiasz?"
  formal: 0.9081
  informal: 0.0919

Text 4: "I feel deep regret and sadness about the situation in international politics."
  formal: 0.9835
  informal: 0.0165
```
