The checksums match so the models appear to be the same. The configurations seem to match up; there isn't as much detailed info for Candle.

# PyTorch

```bash
➜  pytorch git:(master) ✗ cd ~/.cache/huggingface/hub/models--s-nlp--xlmr_formality_classifier/
➜  models--s-nlp--xlmr_formality_classifier ls -la snapshots/*/
lrwxr-xr-x@ - jon 18 May 06:34 config.json -> ../../blobs/f8675475652c3fe67225d990af840701e5a57e4a
lrwxr-xr-x@ - jon 18 May 06:35 model.safetensors -> ../../blobs/66037d963856d6d001f3109d2b3cf95c76bce677947e66f426299c89bc1b58e7
lrwxr-xr-x@ - jon 18 May 11:43 special_tokens_map.json -> ../../blobs/2ea7ad0e45a9d1d1591782ba7e29a703d0758831
lrwxr-xr-x@ - jon 18 May 06:34 tokenizer.json -> ../../blobs/4fffd24ea9482547f219e7e242016fbf4baf5bc6
lrwxr-xr-x@ - jon 18 May 11:43 tokenizer_config.json -> ../../blobs/04075d712fbb8101ab508c10f8606e21b72b1edf
➜  models--s-nlp--xlmr_formality_classifier sha256sum snapshots/*/model.safetensors
66037d963856d6d001f3109d2b3cf95c76bce677947e66f426299c89bc1b58e7  snapshots/d8336084562d7755d9a3e2d5f5d88b9778c6ef7b/model.safetensors
```

```python
print("========== CONFIGURATION ==========")
print(f"Model Name: {model.name_or_path}")
print(f"Model Network: {model}")
print(f"Model Config: {model.config}")
```

```bash
========== CONFIGURATION ==========
Model Name: s-nlp/xlmr_formality_classifier
Model Network: XLMRobertaForSequenceClassification(
  (roberta): XLMRobertaModel(
    (embeddings): XLMRobertaEmbeddings(
      (word_embeddings): Embedding(250002, 768, padding_idx=1)
      (position_embeddings): Embedding(514, 768, padding_idx=1)
      (token_type_embeddings): Embedding(1, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): XLMRobertaEncoder(
      (layer): ModuleList(
        (0-11): 12 x XLMRobertaLayer(
          (attention): XLMRobertaAttention(
            (self): XLMRobertaSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): XLMRobertaSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): XLMRobertaIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): XLMRobertaOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
  )
  (classifier): XLMRobertaClassificationHead(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
    (out_proj): Linear(in_features=768, out_features=2, bias=True)
  )
)
Model Config: XLMRobertaConfig {
  "architectures": [
    "XLMRobertaForSequenceClassification"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "formal",
    "1": "informal"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "formal": 0,
    "informal": 1
  },
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "xlm-roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "output_past": true,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "problem_type": "single_label_classification",
  "torch_dtype": "float32",
  "transformers_version": "4.52.1",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 250002
}
```

# Rust

```Rust
println!("Rust weights filename: {:?}", &weights_filename);
let config: Config = serde_json::from_str(&config)?;
```

```bash
Rust weights filename: "/Users/jon/.cache/huggingface/hub/models--s-nlp--xlmr_formality_classifier/snapshots/d8336084562d7755d9a3e2d5f5d88b9778c6ef7b/model.safetensors"
Config:
  hidden_size: 768
  layer_norm_eps: 0.00001
  attention_probs_dropout_prob: 0.1
  hidden_dropout_prob: 0.1
  num_attention_heads: 12
  position_embedding_type: "absolute"
  intermediate_size: 3072
  hidden_act: Gelu
  num_hidden_layers: 12
  vocab_size: 250002
  max_position_embeddings: 514
  type_vocab_size: 1
  pad_token_id: 1
```


```bash
➜  candle git:(master) ✗ sha256sum /Users/jon/.cache/huggingface/hub/models--s-nlp--xlmr_formality_classifier/snapshots/d8336084562d7755d9a3e2d5f5d88b9778c6ef7b/model.safetensors
66037d963856d6d001f3109d2b3cf95c76bce677947e66f426299c89bc1b58e7  /Users/jon/.cache/huggingface/hub/models--s-nlp--xlmr_formality_classifier/snapshots/d8336084562d7755d9a3e2d5f5d88b9778c6ef7b/model.safetensors
```


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
