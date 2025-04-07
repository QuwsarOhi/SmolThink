import ollama
import json
from ast import literal_eval
import re
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
from webtool.webtool import webtool_def, search_tool, tool_call_extract, tool_parse, remove_think


SYS_PROMPT = '''You are a helpful AI assistant named SmolThink.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> tags:
<tools>
{"type": "function", "function": {"name": "web_search", "description": "Can search the web for infomation which are doubtful/unknown/recent.", "parameters": {"type": "object", "properties": {"search_str": {"type": "string", "description": "The whole question you want to ask.", "required": true}}}}}
</tools>

You first think/plan inside <think></think> tags.
Then for each function call, return a json object with function name and arguments within <tool_call></tool_call> tags.'''

# Ollama debug
# https://stackoverflow.com/questions/78609187/how-to-print-input-requests-and-output-responses-in-ollama-server

# Chat with Ollama and stream responses
def chat_with_ollama(user_input):
    messages = [
        # {"role": "system", "content": "You are a helpful AI assistant named SmolThink. First plan/reason/code/validate inside <think></think> tag and provide final answer to user query inside <answer></answer> tag."},
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_input}
    ]
    
    # First call: Initial chat with streaming enabled
    response_stream = ollama.chat(
        model="SmolThink",
        messages=messages,
        # tools=[webtool_def],
        stream=True,
        options={
            'ctx': 1024*8,
            'temperature': 0.8
        }
    )
    
    collected_response = ""  # Store the final assistant message
    for chunk in response_stream:
        content = chunk["message"].get("content", "")
        print(content, end="", flush=True)  # Stream response in real-time
        collected_response += content
    print(flush=True)

    # # Handle tool calls if present
    try:
        tool_call = tool_call_extract(collected_response)    
        result, urls = search_tool(
            tool_call["arguments"]["search_str"], 
            trim=None,#1024 * 8,
            max_results=2,
        )
        result += f"\n\nUser question: {user_input}\n"
    except Exception:
        print("Cannot process tool_call. Falling back", flush=True)
        return

    print("\n\n" + "-"*50, end="\n\n")
    
    if tool_call:
        messages.append({"role": "assistant", "content": remove_think(collected_response)})
        messages.append({"role": "tool", "content": result})
        # messages.pop(0)
        # messages = [
        #     {"role": "tool", "content": "You are a helpful AI assistant named SmolThink. First plan/reason/code/validate inside <think></think> tag and provide final answer to user query inside <answer></answer> tag."}
        # ] + messages
        # print(json.dumps(messages, indent=1))

        # Send tool response back to Ollama with streaming
        tool_response_stream = ollama.chat(
            model="SmolThink",
            messages=messages,
            # tools=[webtool_def],
            stream=True,
            options={'ctx': 1024*8}
        )

        # Stream the tool response
        for tool_chunk in tool_response_stream:
            content = tool_chunk["message"].get("content", "")
            print(content, end="", flush=True)
    

# Example Usage
while True:
    user_input = input("Chat input: ")
    chat_with_ollama(user_input)
    print(flush=True)
    print("\n\n" + "="*50, end="\n\n")
