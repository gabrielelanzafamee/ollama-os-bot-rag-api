import re
import requests
from bs4 import BeautifulSoup

from llm import LLM
from embedding import Embedding
from enviroment import SYSTEM_PROMPT

class Bot:
    def __init__(self, model_id):
        self.embedding = Embedding("mixedbread-ai/mxbai-embed-large-v1")
        self.llm = LLM(model_id)
        
    def chat(self, messages: list, temperature=0.5, stop=[], top_p=0.95, top_k=0.5, stream=False):
        web_context = self.online_context(messages)
        context = "\n\n".join(map(lambda item: item[2], web_context[:3]))
        
        # add system prompt
        messages.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})

        llm_response = self.llm.chat(messages, temperature, stop, top_p, top_k, stream)
        
        if stream:
            for chunk in llm_response:
                yield chunk['message']['content']
        else:
            return llm_response['messages']['content']
    
    def online_context(self, messages):
        prompt = messages[-1]['content']
        urls = self._extract_urls(prompt)

        if len(urls) == 0:
            return []
        
        pages = self._get_pages(urls)
        pages = "\n\n".join(pages)

        chunk_size = 1024 # in characters
        chunks = [pages[i:i+chunk_size] for i in range(0, len(pages), chunk_size)]
        embeddings_chunks = self.embedding.encode(chunks)
        
        ## prompt without urls
        query = re.sub(r'(https?://\S+)', '', prompt)
        query = self.embedding.transform_query(query)
        query = self.embedding.encode([query])[0]

        similarities = self.embedding.get_similarities(query, embeddings_chunks)
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
    