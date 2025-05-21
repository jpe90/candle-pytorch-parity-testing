from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification

tokenizer = XLMRobertaTokenizerFast.from_pretrained('s-nlp/xlmr_formality_classifier')
model = XLMRobertaForSequenceClassification.from_pretrained('s-nlp/xlmr_formality_classifier')

id2formality = {0: "formal", 1: "informal"}
texts = [
    "I like you. I love you",
    "Hey, what's up?",
    "Siema, co porabiasz?",
    "I feel deep regret and sadness about the situation in international politics.",
]

print("========== TOKENIZATION INFO ==========")
for i, text in enumerate(texts):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    print(f"\nText {i+1}: \"{text}\"")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Token ID to Token Mapping:")
    for id in token_ids:
        if id not in [tokenizer.bos_token_id, tokenizer.eos_token_id]:
            print(f"  {id} → {tokenizer.convert_ids_to_tokens(id)}")
    print("-" * 40)
print("=======================================\n")

encoding = tokenizer(
    texts,
    add_special_tokens=True,
    return_token_type_ids=True,
    truncation=True,
    padding="max_length",
    return_tensors="pt",
)

# dump config
print("========== CONFIGURATION ==========")
print(f"Model Name: {model.name_or_path}")
print(f"Model Network: {model}")
print(f"Model Config: {model.config}")

output = model(**encoding)

print(f"PyTorch logits: {output.logits}")

formality_scores = [
    {id2formality[idx]: score for idx, score in enumerate(text_scores.tolist())}
    for text_scores in output.logits.softmax(dim=1)
]
print("Formality Scores:\n")
for i, (text, scores) in enumerate(zip(texts, formality_scores)):
    print(f"Text {i+1}: \"{text}\"")
    for formality, score in scores.items():
        print(f"  {formality}: {score:.4f}")
    print()
