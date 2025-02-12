import json
import re
from .model import generate_text
import torch

DEBUG = False

class RAG:
    def __init__(self, model, tokenizer, memory):
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory
        self.history = []

    def clear_history(self):
        self.history = []
        return

    def extract_json_from_output(self, text: str) -> dict:
        # Regex that looks for triple backtick with "json" and captures everything in between
        pattern = r"```json\s*({.*?})\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        # If that fails, try to parse any JSON-like content
        # or handle errors gracefully
        return {}

    def get_preamble(self):
        preamble = f"""
You are an AI assistant with expertise regarding medical knowledge. You should be as professional as possible.
If the user asks you something not related to medical knowledge, you MUST politely decline to answer.
        """
        return preamble

    def get_history_as_string(self, history_go_back):
        start_index = max(0, len(self.history) - history_go_back)
        # format last history_go_back question + answers in the chatbot
        conversation_history = "\n\n".join([h for h in self.history[start_index:]])

        if DEBUG:
            print(f"DEBUG: Conversation history string:\n\n {conversation_history}\nDEBUG: End Conversation history string")

        return conversation_history

    def retrieve_from_memory(self, message: str):
        # Get history of chat
        conversation_history = self.get_history_as_string(history_go_back=2)

        # We exploit the LLM to generate the query
        prompt = f"""
Your task is generating a query of TWO/THREE words given a message of the User. The query will be used to
find matches in a vector database to retrieve relevant information.

# Context
## Past conversations history
{conversation_history}

## Current message
The User wrote the following:
<START_USER_MESSAGE>
{message}
<END_USER_MESSAGE>

Please return ONLY the query as a JSON object.

## The JSON must have the following structure:

```json
{{
    "query": // str - The query
}}
```

Now you can generate the JSON. You must generate only the markdown with the JSON. NOTHING ELSE.