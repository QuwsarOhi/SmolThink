import ollama
import json
from ast import literal_eval
import re
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests

# Define available tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Can search the web for infomation which are doubtful/unknown/recent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_str": {
                        "type": "string",
                        "description": "The whole question you want to ask. Has to be complete and informative WH-question.",
                    }
                },
                "required": ["search_str"]
            },
        },
    }
]

def url_content(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove scripts, styles, navs, headers, footers, and typical ad elements
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'noscript', 'iframe']):
        tag.decompose()

    ad_classes = ['advertisement', 'ad', 'adsbygoogle', 'promo', 'banner', 'cookie-banner', 'subscribe']
    for class_name in ad_classes:
        for tag in soup.select(f'.{class_name}, #{class_name}'):
            tag.decompose()

    # Remove all links and their content
    for a_tag in soup.find_all('a'):
        a_tag.decompose()

    # Markdown content building
    markdown_lines = []

    # Process headings and paragraphs
    for element in soup.body.descendants if soup.body else soup.descendants:
        if element.name:
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(element.name[1])
                markdown_lines.append(f"{'#' * level} {element.get_text(strip=True)}\n")
            elif element.name in ['ul', 'ol']:
                for li in element.find_all('li'):
                    markdown_lines.append(f"- {li.get_text(strip=True)}")
            elif element.name == 'p':
                text = element.get_text(strip=True)
                if text:
                    markdown_lines.append(f"{text}\n")

    # Final cleanup: remove empty lines
    markdown_lines = list(filter(lambda x: len(x) > 2, markdown_lines))
    markdown_content = '\n'.join([line for line in markdown_lines if line.strip()])    
    return markdown_content


def search_tool(search_str, full_content=True, max_results=1):
    rets = None
    with DDGS() as ddg:
        rets = list(ddg.text(keywords=search_str, region="wt-wt", max_results=max_results))

    str_rets = ''
    if full_content:
        str_rets = "Use following information to answer user_question:\n\n"
        for i, r in enumerate(rets):
            r = url_content(rets[i]['href'])[:1024*3]
            str_rets += f"# Source {i+1}:\n" + "-"*10 + f"\n\n{r}\n\n\n"
    else:
        str_rets = "Use following information to answer user_question:\n\n"
        for i, r in enumerate(rets):
            str_rets += f"# Source {i+1}:\n" + "-"*10 + f"\n\n{r}\n\n\n"
        
    return str_rets


def tool_parse(tool_call:str):
    ret = None
    try:
        ret = literal_eval(tool_call)
    except:
        pass
    
    try:
        _tool_call = tool_call.replace("'", '"')
        ret = json.loads(_tool_call)
    except:
        pass
    return ret


def tool_call_extract(inp_str:str):
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    tool_calls = pattern.findall(inp_str)
    if tool_calls:
        tool_call = tool_parse(tool_calls[0])
        return tool_call
    return None

# Chat with Ollama and stream responses
def chat_with_ollama(user_input):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant named SmolThink. First plan/reason/code/validate inside <think></think> tag and provide final answer to user query inside <answer></answer> tag."},
        {"role": "user", "content": user_input}
    ]
    
    # First call: Initial chat with streaming enabled
    response_stream = ollama.chat(
        model="SmolThink",
        messages=messages,
        tools=tools,
        stream=True,
        options={'ctx': 1024*8}
    )
    
    collected_response = ""  # Store the final assistant message

    for chunk in response_stream:
        content = chunk["message"].get("content", "")
        print(content, end="", flush=True)  # Stream response in real-time
        collected_response += content

    # Handle tool calls if present
    tool_call = None
    tool_call = tool_call_extract(collected_response)

    # print(tool_call)
    
    if tool_call:
        result = search_tool(tool_call['arguments']['search_str'])
        messages.append({"role": "assistant", "content": collected_response})
        messages.append({"role": "tool", "name": "web_search", "content": result})

        print("\n\n" + "-"*50, end="\n\n")
        print(json.dumps(messages, indent=1))

        # Send tool response back to Ollama with streaming
        tool_response_stream = ollama.chat(
            model="SmolThink",
            messages=messages,
            tools=tools,
            stream=True
        )

        # Stream the tool response
        for tool_chunk in tool_response_stream:
            content = tool_chunk["message"].get("content", "")
            print(content, end="", flush=True)

print("\n")  # Newline for better readability

# Example Usage

while True:
    user_input = input("Chat input: ")
    chat_with_ollama(user_input)
    print()
