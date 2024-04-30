import re
import requests
from bs4 import BeautifulSoup
import ollama
from transformers import AutoModel, AutoTokenizer
from sentence_transformers.util import cos_sim
from typing import Dict
import torch
import numpy as np

from fastapi import FastAPI

class Embedding:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)

    def transform_query(self, query: str) -> str:
        """ For retrieval, add the prompt for query (not for documents).
        """
        return f'Represent this sentence for searching relevant passages: {query}'

    def pooling(self, outputs: torch.Tensor, inputs: Dict,  strategy: str = 'cls') -> np.ndarray:
        if strategy == 'cls':
            outputs = outputs[:, 0]
        elif strategy == 'mean':
            outputs = torch.sum(
                outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])
        else:
            raise NotImplementedError
        return outputs.detach().cpu().numpy()

    def encode(self, docs):
        inputs = self.tokenizer(docs, padding=True, return_tensors='pt')
        for k, v in inputs.items():
            inputs[k] = v
        outputs = self.model(**inputs).last_hidden_state
        embeddings = self.pooling(outputs, inputs, 'cls')
        return embeddings

    def get_similarities(self, embeddings):
        # similarities = cos_sim(embeddings[0], embeddings[1:])
        similarities = []

        for i in range(1, len(embeddings)):
            similarity = cos_sim(embeddings[0], embeddings[i])
            similarities.append((i, similarity))

        return similarities
    

class LLM:
    def __init__(self, model_id):
        self.model_id = model_id

    def chat(self, messages, temperature=0.5, stop=[], top_p=0.95, top_k=0.5, stream=False):
        return ollama.chat(
            model=self.model_id,
            messages=messages,
            options={
                'top_p': top_p,
                'top_k': top_k,
                'temperature': temperature,
            },
            stream=stream
        )

class Bot:
    def __init__(self, model_id):
        self.embedding = Embedding("mixedbread-ai/mxbai-embed-large-v1")
        self.llm = LLM(model_id)
        
    def chat(self, messages, temperature=0.5, stop=[], top_p=0.95, top_k=0.5, stream=False):
        web_context = self.online_context(messages)
        print(web_context[0])

        context = "\n\n".join(map(lambda item: item[2], web_context[:3]))
        messages[0]['content'] = f"{messages[0]['content']}\n\n##Context\n{context}"

        print(messages)

        return self.llm.chat(messages, temperature, stop, top_p, top_k, stream)
    
    def online_context(self, messages):
        prompt = messages[-1]['content']
        urls = self._extract_urls(prompt)
        pages = self._get_pages(urls)
        pages = "\n\n".join(pages)

        chunk_size = 1024 # in characters
        chunks = [pages[i:i+chunk_size] for i in range(0, len(pages), chunk_size)]
        embeddings_chunks = np.array(list(map(self.embedding.encode, chunks))[0])
        
        ## prompt without urls
        prompt = re.sub(r'(https?://\S+)', '', prompt)
        prompt = self.embedding.transform_query(prompt)

        query = self.embedding.encode([prompt])
        embeddings = np.concatenate((query, embeddings_chunks), axis=0)

        similarities = self.embedding.get_similarities(embeddings)
        similarities = list(map(lambda x: (x[0], x[1].item(), chunks[x[0]-1]), similarities))
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

        return similarities

    def _get_pages(self, urls):
        pages = []

        for url in urls:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            for p in soup.find_all('p'):
                pages.append(p.get_text())

        return pages

    def _extract_urls(self, text):
        return re.findall(r'(https?://\S+)', text)
    
    
bot = Bot("phi3")


while True:
    stream = bot.chat([
        { "role": "system", "content": "You are an AI assistant. Called Bot" },
        { "role": "user", "content": input("> ") },
    ], stream=True, temperature=0.1, top_p=0.95, top_k=0.4)

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

    print("\n\n")
