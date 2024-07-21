import duckduckgo_search
from trafilatura import fetch_url, extract
from duckduckgo_search import DDGS
from prompt_template import *
import ollama
import re
import json


def web_search(search_query, get_web_contents=True):
    results = DDGS().text(search_query, max_results=1)
    sdata = []

    for result in results:
        web = fetch_url(result['href'])
        sdata.append({'title': result['title'], 'content': extract(web)[:4000]})

    # results = json.dumps(results, indent=1)
    return sdata


# https://github.com/ollama/ollama-python/blob/main/examples/async-chat-stream/main.py
# https://github.com/ollama/ollama/blob/main/docs/api.md#request-raw-mode
# https://github.com/ollama/ollama/blob/main/docs/modelfile.md
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
            'num_ctx': 4000,
            'temperature': 0.,
            'top_p': 0.0
        },
    )

def get_llm_response(prompt):
    ret = ''
    print('', flush=True)
    for token in generate(prompt):
        print(token['response'], end='', flush=True)
        ret += token['response']
    return ret


def extract_response(response):
    if '[TOOL_CALLS]' in response: #or 'web_search' in response:
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
        "system": SYSTEM_PROMPT,
        "prompt": question,
        "tools": TOOLS,
        # "toolcalls": '[TOOL_CALL]'
    }

while True:
    # question = "What is the recent news of Bangladesh?"
    question = '[QUESTION] ' + input("\n\nAsk something: ") + ' [QUESTION]'
    init_prompt = build_init_prompt(question)
    prompt = prompt_builder(init_prompt)
    print(prompt, end='----\n')

    memory = push_memory(init_prompt)
    response = get_llm_response(prompt)
    output = extract_response(response)

    if output:
        memory = {
            'messages': memory['messages'],
            'role': 'assistant',
            'toolcalls': response.replace('[TOOL_CALLS]', '').strip(), 
        }
        memory = push_memory(memory)
        memory = {
            'messages': memory['messages'],
            'role': 'tool',
            'toolres': output,
        }
        prompt = prompt_builder(memory)
        print("\n------------------")
        print(prompt, end='\n--------------\n')
        response = get_llm_response(prompt)
    else:
        memory = {
            'messages': memory['messages'],
            'role': 'assistant',
            'response': response
        }
        memory = push_memory(memory)