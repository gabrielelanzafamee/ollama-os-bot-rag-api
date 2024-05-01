import ollama

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