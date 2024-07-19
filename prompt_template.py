# https://ollama.com/library/mistral/blobs/491dfa501e59
# 
# {{- if .Messages }}
#     {{- range $index, $_ := .Messages }}
#         {{- if eq .Role "user" }}
#             {{- if and (eq (len (slice $.Messages $index)) 1) $.Tools }}
#                 [AVAILABLE_TOOLS] {{ $.Tools }}[/AVAILABLE_TOOLS]
#             {{- end }}
#             [INST] 
#             {{ if and $.System (eq (len (slice $.Messages $index)) 1) }}
#                 {{ $.System }}
#             {{ end }}
#             {{ .Content }}[/INST]
#        
#         {{- else if eq .Role "assistant" }}
#             {{- if .Content }} 
#                 {{ .Content }}
#             {{- else if .ToolCalls }}
#                 [TOOL_CALLS] [
#                     {{- range .ToolCalls }}
#                     {"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}} {{- end }}]
#             {{- end }}</s>
#        
#         {{- else if eq .Role "tool" }}
#             [TOOL_RESULTS] {"content": {{ .Content }}} [/TOOL_RESULTS] 
#         {{- end }}
#   
#     {{- end }}
#
# {{- else }}[INST] {{ if .System }}{{ .System }}
# {{ end }}{{ .Prompt }}[/INST]
# {{- end }} {{ .Response }}
# {{- if .Response }}</s>
# {{- end }}

def prompt_builder(data):
    prompt = ""
    for idx, message in enumerate(data.get('messages')):
        if message.get('role') == 'user':
            if message.get('tools') and idx == 0:
                prompt += f"[AVAILABLE_TOOLS] {message['tools']} [/AVAILABLE_TOOLS]\n"
            prompt += "[INST] "
            if message.get('system') and idx == 0:
                prompt += f"{message['system']}\n"
            prompt += f"{message.get('content')} [/INST]\n"

        elif message.get('role') == 'assistant':
            if message.get('content'):
                prompt += message['content']
            elif message.get('toolcalls'):
                prompt += f"[TOOL_CALLS] [\n{message['toolcalls']}\n]"
            prompt += "</s>\n"

        elif message.get('role') == 'tool':
            prompt += f"[TOOL_RESULTS] {message['content']} [/TOOL_RESULTS]\n"
    
    prompt += "\n"

    if data.get('tools'):
        prompt += f"[AVAILABLE_TOOLS] {data['tools']} [/AVAILABLE_TOOLS]\n"
    prompt += "[INST] "
    if data.get('system'):
        prompt += data['system'] + "\n"
    if data.get('prompt')
        prompt += data.get['prompt'] + " [/INST]\n"
    
    if data.get('response'):
        prompt += data['response'] + " </s>"
    if data.get('toolcalls'):
        pass
    if data.get('toolresult'):
        pass

    return prompt


if __name__ == '__main__':    
    context = {
        "messages": [
            {"role": "user", "content": "User message", "system": 'system message', "tools": "some_tool"},
            {"role": "assistant", "content": "Assistant response", },
            {"role": "assistant", "toolcalls": [{"Function": {"Name": "tool_func", "Arguments": ["arg1", "arg2"]}}]},
            {"role": "tool", "content": "Tool result"},
        ],
        "prompt": "Initial prompt",
        "response": "Final response",
        "system": "system prompt",
        "tools": "tools",
        "tool_result": "result"
    }

    print(prompt_builder(context))