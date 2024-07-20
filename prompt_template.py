# https://ollama.com/library/mistral/blobs/491dfa501e59

# SYSTEM_PROMPT = "You are a helpful AI named Jarvis. You can perform function calling to search the web and give precise and short answers."

SYSTEM_PROMPT = "You are a helpful AI assistant. Perform function calling to search the web, visit urls. Give precise answers by visiting urls with function calling. Do not give reference to websites or urls."

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Perform web search",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Query for web search",
                    }
                },
                "required": ["search_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visit_url",
            "description": "Visit a particular url",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to browse",
                    }
                },
                "required": ["url"],
            },
        },
    }
]

def push_memory(data):
    messages = data.get('messages', [])
    new_msg = {}

    if data['role'] == 'user':
        new_msg = {
            'role': 'user',
            'prompt': data['prompt'],
        }
        if data.get('system'):
            new_msg['system'] = data['system']
        if data.get('tools'):
            new_msg['tools'] = data['tools']

    elif data['role'] == 'assistant':
        new_msg['role'] = 'assistant'
        if data.get('response'):
            new_msg['response'] = data['response']
        elif data.get('toolcalls'):
            new_msg['toolcalls'] = data['toolcalls']

    elif data['role'] == 'tool':
        new_msg = {
            'role': 'tool',
            'toolres': data['toolres']
        }

    return {"messages": messages + [new_msg]}


def prompt_builder(data):
    prompt = ""
    for idx, message in enumerate(data.get('messages')):
        if message.get('role') == 'user':
            if message.get('tools') and idx == 0:
                prompt += f"[AVAILABLE_TOOLS] {message['tools']} [/AVAILABLE_TOOLS]\n"
            prompt += "[INST] "
            if message.get('system') and idx == 0:
                prompt += f"{message['system']}\n"
            prompt += f"{message.get('prompt')} [/INST]\n"

        elif message.get('role') == 'assistant':
            if message.get('response'):
                prompt += message['response']
            elif message.get('toolcalls'):
                prompt += f"[TOOL_CALLS] \n{message['toolcalls']}\n"
            prompt += "</s>\n"

        elif message.get('role') == 'tool':
            prompt += f"[TOOL_RESULTS] {message['toolres']} [/TOOL_RESULTS]\n"
    
    prompt += "\n"
    if data.get('role') == 'user' and data.get('tools'):
        prompt += f"[AVAILABLE_TOOLS] {data['tools']} [/AVAILABLE_TOOLS]\n"
    if data.get('role') == 'user':
        prompt += "[INST] "
    if data.get('system'):
        prompt += data['system'] + "\n"
    if data.get('prompt'):
        prompt += data['prompt'] + " [/INST]\n"
    if data.get('response'):
        prompt += data['response'] + " </s>\n"
    if data.get('toolcalls'):
        prompt += data['toolcalls'] + " </s>\n"
    if data.get('toolres'):
        prompt += f"[TOOL_RESULTS] {data['toolres']} [/TOOL_RESULTS]\n"

    return prompt


if __name__ == '__main__':    
    context = {
        "messages": [
            {"role": "user", "prompt": "User message", "system": 'system message', "tools": "some_tool"},
            {"role": "assistant", "response": "Assistant response", },
            {"role": "assistant", "toolcalls": [{"Function": {"Name": "tool_func", "Arguments": ["arg1", "arg2"]}}]},
            {"role": "tool", "toolres": "Tool result"},
        ],
        "prompt": "Initial prompt",
        "response": "Final response",
        "system": "system prompt",
        "tools": "tools",
        "tool_result": "result",
        "role": "user"
    }

    print(prompt_builder(context))
    push_memory(context)