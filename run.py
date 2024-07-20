import duckduckgo_search
from trafilatura import fetch_url, extract
from duckduckgo_search import DDGS
from prompt_template import *
import ollama
import re
import json


def web_search(search_query):
    results = DDGS().text(search_query, max_results=2)
    # results = json.dumps(results, indent=1)
    return results


# https://github.com/ollama/ollama-python/blob/main/examples/async-chat-stream/main.py
def generate(prompt):
    # https://github.com/ollama/ollama-python/blob/00eafed0faa5dea6879a8eb3229c7a8f2439abb4/ollama/_types.py#L93
    return ollama.generate(
        model = 'mistral',
        # system = system,
        # Raw is set to true to feed the question as needed
        raw=True,
        prompt=prompt,
        stream = True,
        # Number of seconds to keep the connection alive
        # keep_alive=60*60,
        options = {
            # 'stop': ['</s>', '}}]'],
            'top_k': 1,
            #'cache': False,
            # 'tfs_z': 2.0,
            'num_ctx': 4048,
            'temperature': 0.,
            'top_p': 0.0
        },
    )


def get_llm_response(prompt):
    ret = ''
    for token in generate(prompt):
        print(token['response'], end='', flush=True)
        ret += token['response']
    return ret


def extract_response(response):
    if '[TOOL_CALLS]' in response or 'web_search' in response:
        response = response.replace('[TOOL_CALLS]', '')
        pattern = r'\[(.*?)\]'
        fcall = re.findall(pattern, response)[0]
        fcall = fcall.replace("'", "\"")
        data = json.loads(fcall)

        if data.get('name') == 'web_search':
            squery = data.get('arguments').get('search_query')
            return web_search(squery)
        return None


# web = fetch_url("https://docs.mistral.ai/capabilities/function_calling/#step-1-user-specify-tools-and-query")
# text = extract(web)

def build_init_prompt(question):
    return {
        "messages": [],
        "role": "user",
        # "system": SYSTEM_PROMPT,
        "prompt": question,
        "tools": TOOLS, #json.dumps(TOOLS, indent=1).split()),
        # "toolres": ''
    }

question = "What is the weather today?"

init_prompt = build_init_prompt(question)
prompt = prompt_builder(init_prompt)
print(prompt)

response = get_llm_response(prompt)
output = extract_response(response)

memory = push_memory(init_prompt)
memory = {
    'messages': memory['messages'],
    'role': 'assistant',
    'toolcalls': response, 
}
memory = push_memory(memory)
memory = {
    'messages': memory['messages'],
    'role': 'tool',
    'toolres': output,
}

# print("\memory")
# print(memory)

print("\n*")
print(json.dumps(memory, indent=1))
print("*")

prompt = prompt_builder(memory)
print("\n*")
print(prompt)

response = get_llm_response(prompt)