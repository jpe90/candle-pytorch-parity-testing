import argparse
import time
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModel, BertModel
import numpy as np
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Embed text using Hugging Face transformers')
    parser.add_argument('--prompt', type=str, help='Prompt to embed')
    parser.add_argument('--normalize_embeddings', type=bool, default=False, 
                        help='Whether to normalize embeddings')
    return parser.parse_args()

def build_model_and_tokenizer():
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    device = torch.device("cpu")
    model = model.to(device)
    return model, tokenizer, device

def normalize_l2(embeddings):
    return F.normalize(embeddings, p=2, dim=1)

def main():
    args = parse_args()
    model, tokenizer, device = build_model_and_tokenizer()
    
    if args.prompt:
        prompt = args.prompt
        print(f"prompt: {prompt}")
        encoded_input = tokenizer(prompt, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        # start_time = time.time()
        
        with torch.no_grad():
            outputs = model(**encoded_input)
            embeddings = outputs.last_hidden_state
        
        print(f"Raw output shape: {embeddings.shape}")
        
        if args.normalize_embeddings:
            embeddings = normalize_l2(embeddings)
            
        print(embeddings)
        # print(f"Took {time.time() - start_time:.4f} seconds")
        
    else:
        sentences = [
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
            "The new movie is awesome",
            "The cat plays in the garden",
            "A woman watches TV",
            "The new movie is so great",
            "Do you like pizza?",
        ]
        n_sentences = len(sentences)
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        print(f"running inference on batch {encoded_input['input_ids'].shape}")
        
        with torch.no_grad():
            outputs = model(**encoded_input)
            embeddings = outputs.last_hidden_state
        
        print(f"generated embeddings {embeddings.shape}")
        
        n_tokens = embeddings.shape[1]
        embeddings = embeddings.sum(dim=1) / n_tokens
        
        if args.normalize_embeddings:
            embeddings = normalize_l2(embeddings)
            print(f"normalized embeddings {embeddings.shape}")
        
        print(f"pooled embeddings {embeddings.shape}")
        
        similarities = []
        for i in range(n_sentences):
            for j in range(i + 1, n_sentences):
                cosine_similarity = F.cosine_similarity(embeddings[i].unsqueeze(0), 
                                                       embeddings[j].unsqueeze(0)).item()
                similarities.append((cosine_similarity, i, j))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        for score, i, j in similarities[:5]:
            print(f"score: {score:.2f} '{sentences[i]}' '{sentences[j]}'")

if __name__ == "__main__":
    main()
