from io import BytesIO
import requests
import json
import re
from copy import deepcopy
from ast import literal_eval

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


def tool_parse(tool_call: str):
    """
    Parses tool call in two different formats:
    {'function_name': 'fun1', 'arguments': {...}}
    {"function_name": "fun1", "arguments": {...}}
    """

    ret = None
    try:
        ret = literal_eval(tool_call)
    except Exception:
        pass

    _tool_call = tool_call.replace("'", '"')
    ret = json.loads(_tool_call)
    return ret


def tool_call_extract(inp_str: str):
    """
    Extracts tool call from format:
    <tool_call>
    JSON tool call
    </tool call>
    """
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    tool_calls = pattern.findall(inp_str)
    if tool_calls:
        tool_call = tool_parse(tool_calls[0])
        return tool_call
    return None


def remove_think(inp_str: str):
    """Removes 'think' tokens from LLM generated outputs
    LLM would usually generate the following response pattern:
    <think>
    Let's think step by step...
    </think>
    <tool_call>
    JSON tool call
    <tool_call>
    """
    inp_str = deepcopy(inp_str)
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    thinks = pattern.findall(inp_str)
    for think in thinks:
        inp_str = inp_str.replace(think, "")
    return inp_str


def replace_short_lines(text, new_line="\n"):
    # Split the text into lines
    lines = text.splitlines()
    # Iterate through each line and replace short lines
    updated_lines = [line if len(line) >= 3 else new_line for line in lines]
    # Join the updated lines back into a single string
    return "\n".join(updated_lines)


def docling_cleanup(input_str):
    # <!-- image --> tag cleanup
    input_str = input_str.replace("<!-- image -->", "")
    # Lines with empty spaces
    input_str = replace_short_lines(input_str, "\n")
    # clean excessive newlines
    _cnt = 0
    ret_str = ""
    for c in input_str:
        if c == "\n":
            _cnt += 1
            if _cnt > 2:
                continue
            else:
                ret_str += c
        else:
            _cnt = 0
            ret_str += c
    return ret_str


def url_content(url):
    headers = {"User-Agent": "Mozilla/5.0"}
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

    str_rets = ""
    urls = []
    web_contents = []
    i = -1
    while len(urls) < max_results and i + 1 < len(rets):
        i += 1
        try:
            print("Parsing url:", rets[i]["href"], flush=True)
            web_content = url_content(rets[i]["href"])
            web_content = web_content.strip()
            if web_content == "":
                continue

            web_contents.append(web_content)
            web_content = web_content[:trim] + " (truncated)..."

            content = f"\n# Source {len(urls) + 1}:"
            content += "\n" + "-" * len(content) + f"\n\n{web_content}\n\n"
            str_rets += content
            urls.append(rets[i]["href"])

        except:
            continue

    return str_rets, urls, web_contents


def recursive_chunker(text, min_char_len, stop_token):
    # Good blog to read: https://ai4nerds.github.io/blog/rag/Different%20types%20of%20Chunking.html
    # Base case: if the text is shorter than the minimum length, return an empty list
    if len(text) < min_char_len:
        return []
    # Check if the stop token is present in the text
    stop_index = text.find(stop_token)
    # If the stop token is not found, treat the whole text as one chunk
    if stop_index == -1:
        if len(text) >= min_char_len:
            return [text]
        else:
            return []

    # If the stop token is found, split the text at the stop token
    chunk = text[:stop_index].strip()
    remaining_text = text[stop_index + len(stop_token) :].strip()
    # If the chunk is valid (meets the minimum length), include it in the result
    chunks = []
    if len(chunk) >= min_char_len:
        chunks.append(chunk)
    # Recursively process the remaining text
    return chunks + recursive_chunker(remaining_text, min_char_len, stop_token)


def chunk_relevance(model, tokenizer, question, chunks):
    PROMPT = """<empty_output><|im_start|>system
You will be given a 'content_chunk' and a 'question'. You have to reply 'yes', if the 'content_chunk' is relevent to the 'question'. Otherwise reply 'no'.<|im_end|>
<|im_start|>user

Here is the 'content_chunk':
----------------------------
{chunk}

---

Here is the 'question':
-----------------------
{question}

---

Now, is the 'content_chunk' relevant based on the 'question'?
<|im_end|>
<|im_start|>assistant
Based on the given 'content_chunk' and 'question' my answer is: '"""

    relevance = []
    for idx, chunk in enumerate(chunks):
        prompt = PROMPT.format(chunk=chunk, question=question)
        input_ids = tokenizer(prompt, return_tensors="pt")
        ret = model.generate(
            # streamer=streamer,
            max_new_tokens=1,
            do_sample=False,
            # temperature=0.9,
            input_ids=input_ids["input_ids"].to("mps"),
            attention_mask=input_ids["attention_mask"].to("mps"),
        )
        input_token_len = input_ids["input_ids"].shape[-1]
        ret = tokenizer.decode(ret[0][input_token_len:], skip_special_tokens=True)
        print("LLM RET:", ret)
        relevance.append("yes" in ret)
    return relevance


if __name__ == "__main__":
    from transformers import TextStreamer, AutoModelForCausalLM, AutoTokenizer
    import torch

    PATH = "weights/SmolThink-360M-sft-v2/checkpoint-50400"
    model = AutoModelForCausalLM.from_pretrained(
        PATH,
        device_map="mps",
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # 'sdpa'
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
        tie_word_embeddings=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(PATH)

    question = "Write a python code to do DFS in adjacancy list"  
    # "Population of Palestine"
    str_rets, urls, web_contents = search_tool(question, trim=None, max_results=1)

    # streamer = TextStreamer(tokenizer, skip_prompt=False)
    chunks = recursive_chunker(web_contents[0], min_char_len=256, stop_token="\n\n")
    relevance = chunk_relevance(model, tokenizer, question, chunks)

    print("Total chunks:", len(chunks))
    for c, r in zip(chunks, relevance):
        print(c)
        print(r)
        print("-+" * 20)
