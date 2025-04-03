from io import BytesIO
import requests

from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

from duckduckgo_search import DDGS
# import ollama


# summarize_template = \
# """<|system|>
# You are a helpful assistant. You will be given a web content in markdown format. You have to provide a summary of the web content.
# You have to summarize the web content inside 'summary' tag.
# For example: <summary> your summarization content in markdown </summary><|end|>
# <|user|>
# {web_content}<|end|>
# <|assistant|>
# Sure! Here is the summarized version of the provided content:
# <summary>"""

# def web_content_summarize(web_content):
#     prompt = summarize_template.format(web_content=web_content)
#     # print("Question:", data['question'], flush=True)

#     stream = ollama_infr(prompt=prompt, model='deepseek-r1:7b', temperature=0.5)
#     model_res = '<summary>\n'
#     n_tokens = 0

#     for part in stream:
#         print(part['response'], sep='', end='', flush=True)
#         model_res += part['response']
#         n_tokens += 1

#         if n_tokens > 4000:
#             break

#     summary = re.findall(r"<summary>(.*?)</summary>", model_res, re.DOTALL)#[0].strip()    
#     if summary:
#         return summary[0].strip()
#     return ''

webtool_def = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Can search the web for infomation which are doubtful/unknown/recent.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_str": {
                    "type": "string",
                    "description": "The whole question you want to ask.",
                    "required": True,
                }
            },
        },
    },
}


def replace_short_lines(text, new_line='\n'):
    # Split the text into lines
    lines = text.splitlines()
    # Iterate through each line and replace short lines
    updated_lines = [line if len(line) >= 3 else new_line for line in lines]
    # Join the updated lines back into a single string
    return '\n'.join(updated_lines)

def docling_cleanup(input_str):
    # <!-- image --> tag cleanup
    input_str = input_str.replace('<!-- image -->', '')
    # Lines with empty spaces
    input_str = replace_short_lines(input_str, '\n')
    # clean excessive newlines
    _cnt = 0
    ret_str = ''
    for c in input_str:
        if c == '\n':
            _cnt += 1
            if _cnt > 2: continue
            else: ret_str += c
        else:
            _cnt = 0
            ret_str += c
    return ret_str

def url_content(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    ascii_text = response.text.encode("ascii", "ignore")
    in_doc = InputDocument(
        path_or_stream=BytesIO(ascii_text),
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="duck.html",
    )

    backend = HTMLDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(ascii_text))
    dl_doc = backend.convert()
    return docling_cleanup(dl_doc.export_to_markdown())


def search_tool(search_str, trim=4096, max_results=1):
    rets = None
    with DDGS() as ddg:
        rets = list(ddg.text(keywords=search_str, region="wt-wt", max_results=7))

    str_rets = ''
    web_source = []
    i = -1
    while len(web_source) < max_results and i+1 < len(rets):
        i += 1
        try:
            print("Parsing url:", rets[i]['href'], flush=True)
            web_content = url_content(rets[i]['href'])
            web_content = web_content.strip()
            if web_content == '': continue
            web_content = web_content[:trim] + " (truncated)..."

            # web_content = web_content_summarize(web_content=web_content)
            content = f"\n# Source {len(web_source)+1}:"
            content += "\n" + "-" * len(content) + f"\n\n{web_content}\n\n"
            str_rets += content
            web_source.append(rets[i]['href'])
        except Exception as E:
            continue
        
    return str_rets, web_source

# print(search_tool('current methods used by scientists to improve predictions of climate change and its environmental impact', max_results=1))