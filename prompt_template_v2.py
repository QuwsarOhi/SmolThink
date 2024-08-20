from typing import List, Dict
import ollama

# https://github.com/ollama/ollama-python/blob/main/examples/async-chat-stream/main.py
# https://github.com/ollama/ollama/blob/main/docs/api.md#request-raw-mode
# https://github.com/ollama/ollama/blob/main/docs/modelfile.md
def generate(prompt):
    # https://github.com/ollama/ollama-python/blob/00eafed0faa5dea6879a8eb3229c7a8f2439abb4/ollama/_types.py#L93
    # print(f"PROMPT: ----------------\n${prompt}$\n----------------")
    return ollama.generate(
        # model = 'mistral:7b-instruct-q8_0',
        model = 'mistral',
        raw=True,
        prompt=prompt,
        stream = True,
        # keep_alive=60*60,         # Number of seconds to keep the connection alive
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

# prompt dicts:
# {'role': ['user', 'assistant', 'system'], 'content': str}
# {'role': ['tool_def', 'tool_call'], 'content': dict}
# {'role': []}

def prompt_format(prompts: List[Dict]):
    prompt_str = "<s>"
    prev_role = None
    for prompt in prompts:
        if prompt['role'] in ['system', 'user']:
            # Add end/start token
            if prev_role in ['tool_call', 'assistant']:
                prompt_str += "</s>\n<s>"
            # Token for system/user prompt
            if prev_role not in ['system', 'user']:
                prompt_str += "[INST] "
            # Seperate system prompt
            if prompt['role'] == 'system':
                prompt_str += prompt['content'] + "\n\n"
            else:
                prompt_str += prompt['content']

        elif prompt['role'] in ['tool_def']:
            # Add ending token for system/user
            if prev_role in ['system', 'user']:
                prompt_str += " [/INST]\n"
            prompt_str += f"[AVAILABLE_TOOLS] {prompt['content']} [/AVAILABLE_TOOLS] "
        elif prompt['role'] in ['tool_call']:
            # Add ending token for system/user
            if prev_role in ['system', 'user']:
                prompt_str += " [/INST]\n"
            prompt_str += f" [TOOL_CALLS] {prompt['content']} [/TOOL_CALLS]"
        elif prompt['role'] in ['assistant']:
            # Add ending token for system/user
            if prev_role in ['system', 'user']:
                prompt_str += " [/INST]\n"
            prompt_str += prompt['content']

        prev_role = prompt['role']
    
    # if prev_role in ['assistant']
    if prev_role in ['system', 'user'] and not prompts[-1].get('lead', False):
        prompt_str += ' [/INST]\n'
    
    return prompt_str


# test_prompts = [
#     {'role': 'tool_def', 'content': {'f': '(xyz)'}},
#     {'role': 'system', 'content': 'This is a system prompt.'},
#     {'role': 'user', 'content': 'This is a user prompt.'},
#     {'role': 'tool_call', 'content': 'tool calling mechanism'},
#     {'role': 'user', 'content': 'Something user said.'},
#     {'role': 'assistant', 'content': 'This is a assistant reply'},
#     # {'role': 'tool_call', 'content': 'another tool calling mechanism'},
#     {'role': 'user', 'content': 'what is LLM?'}
# ]


# test_prompts = [
#     {'role': 'user', 'content': '''You are a helpful code assistant. Your task is to generate a valid JSON object based on the given information. So for instance the following:
# name: John
# lastname: Smith
# address: #1 Samuel St.
# would be converted to:'''},
#     {'role': 'assistant', 'content': '''{
# "address": "#1 Samuel St.",
# "lastname": "Smith",
# "name": "John"
# }'''},
#     {'role': 'user', 'content': '''name: Ted
# lastname: Pot
# address: #1 Bisson St.'''}
# ]

prompt = prompt_format(test_prompts)
print(prompt)
ret = get_llm_response(prompt)

print("**" + ret + "**")