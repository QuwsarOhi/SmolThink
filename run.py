import duckduckgo_search
from trafilatura import fetch_url, extract
import ollama

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
            # 'stop': stopwords,
            'top_k': 1,
            #'cache': False,
            'tfs_z': 2.0,
            'num_ctx': 256,
            'temperature': 0.0,
            'top_p': 0.0
        },
    )

# web = fetch_url("https://docs.mistral.ai/capabilities/function_calling/#step-1-user-specify-tools-and-query")
# text = extract(web)

# for idx, line in enumerate(text.split('\n')):
#     print(idx+1, line)

for token in generate("hi"):
    print(token['response'], end='', flush=True)