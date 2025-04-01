# %%
from datasets import load_dataset
import ollama
import random
import json
import re
from bs4 import BeautifulSoup
import requests
from duckduckgo_search import DDGS

# ds = load_dataset("neural-bridge/rag-dataset-12000")['train']

# %%
# Phi3 templates

search_query_template = \
"""<|system|>
You are a helpful assistant. You will be asked a question. Imagine you do not know the answer of the question. As a result, you want to search the web to find the answer. For the given user question, write a search question that could be used in search engine to find the answer to the question. The search string that you would produce should be inside <search tag>. 
For example: <search> your search string </search><|end|>
<|user|>
{query}<|end|>
<|assistant|>
Sure! Here is the one short search string that I would use to search on the web:
<search>"""


question_gen_template = \
"""<|system|>
You are a helpful assistant who is very good at generating question. The user would give you a topic, you have to come up with a question so that the question can be used to train a good quality LLM.<|end|>
<|user|>
Generate question in topic of: {topic_name}

Produce the question inside "question" tag. Example: <question> your generated question </question><|end|>
<|assistant|>
Sure! Here is the question on topic {topic_name}:
<question>"""

# Ref: https://huggingface.co/learn/cookbook/en/llm_judge
llm_judge_search_prompt = \
"""<|system|>
You will be given a topic, user_question, and search_string triplet.
Your task is to provide a 'total rating' scoring how well the search_string alings the user concerns expressed in the user_question.
Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpful at all, and 4 means that the system_answer completely and helpfully addresses the user_question.

Here is the scale you should use to build your answer:
1: The system_question is not on topic: completely irrelevant to the topic, or very partial
2: The system_question is mostly not helpful: has grammatical error or seems to be incomplete
3: The search_string is mostly helpful: provides concise prompt that can be used in web search engine
4: The search_string is excellent: relevant, direct, and addresses important concerns raised in the system_question

Use the following rubrics to award the points:
- Award 1 point if the user_question is related to the topic.
- Give 1 additional point if the system_question is clear and precise.
- Provide 1 further point if the search_str is concise.
- One final point should be awarded if the search_str is direct and addresses important concerns raised in system_question.

Provide your feedback as follows:

Feedback:::
Evaluation: <eval> (your rationale for the rating, as a text) </eval>
Total rating: <rating> (your rating, as a number between 1 and 4) </rating>

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.<|end|>
<|user|>
Now here are the question and answer.

Topic: {topic}
Question: {question}
Search string: {search_str}

Provide your feedback. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.<|end|>
<|assistant|>
Feedback:::
Evaluation: <eval>"""


llm_judge_prompt = \
"""<|system|>
You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpful at all, and 4 means that the system_answer completely and helpfully addresses the user_question.

Here is the scale you should use to build your answer:
1: The system_answer is terrible: completely irrelevant to the question asked, or very partial
2: The system_answer is mostly not helpful: misses some key aspects of the question
3: The system_answer is mostly helpful: provides support, but still could be improved
4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question

Use the following rubrics to award the points:
- Award 1 point if the answer is related to the question.
- Give 1 additional point if the answer is clear and precise.
- Provide 1 further point if the answer is true.
- One final point should be awarded if the answer provides additional resources to support the user.


Provide your feedback as follows:

Feedback:::
Evaluation: <eval> (your rationale for the rating, as a text) </eval>
Total rating: <rating> (your rating, as a number between 1 and 4) </rating>

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.<|end|>
<|user|>
Now here are the question and answer.

Question: {question}
Answer: {answer}

Provide your feedback. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.<|end|>
<|assistant|>
Feedback:::
Evaluation: <eval>"""

# %%
# https://github.com/ollama/ollama-python/blob/main/examples/async-chat-stream/main.py
def ollama_infr(prompt, extra_stops=[], model='phi3.5:latest', temperature=0.7):
    # https://github.com/ollama/ollama-python/blob/00eafed0faa5dea6879a8eb3229c7a8f2439abb4/ollama/_types.py#L93
    return ollama.generate(
        model = model,
        # system = system,
        # Raw is set to true to feed the question as needed
        raw=True,
        prompt = prompt,
        stream = True,
        # Number of seconds to keep the connection alive
        keep_alive=60*60,
        options = {
            'stop': [
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
            ] + extra_stops,
            'temperature': temperature,
            # 'top_k': 1,
            'cache': False,
            # 'tfs_z': 2.0,
            'num_ctx': 6000,
            # 'temperature': 0.0,
            # 'top_p': 0.0
        },
    )

DATA_PATH = "datasets/search_data.jsonl"
def write_jsonl(data: dict):
    with open(DATA_PATH, "a") as f:
        f.write(json.dumps(data) + "\n")


def extract_rating(input_str:str):
    data = input_str.split('<rating>')[1]
    data = data.strip()

    try:
        return float(data)
    except:
        return None

# %%
deepseek_r1 = '''You are a helpful AI. You will be given a question and a context. You have to first think (in 150 words) and then come up with the final answer based on the question and user-provided context

<｜User｜>User question: {question}

Use the following content to answer user question:
{context}

<｜end▁of▁sentence｜>
<｜Assistant｜>
<think>
I will finish thinking in at most 150 words. Now let's think. '''

def r1_response(question, context):    
    prompt = deepseek_r1.format(question=question, context=context)
    # print("Question:", data['question'], flush=True)

    stream = ollama_infr(prompt=prompt, model='deepseek-r1:7b', temperature=0.5)
    model_res = '<think>\n'
    n_think_tokens = 0
    think_finished = False

    for part in stream:
        print(part['response'], sep='', end='', flush=True)
        model_res += part['response']

        if not think_finished and '</think>' in model_res:
            think_finished = True
        if not think_finished:
            n_think_tokens += 1

        if n_think_tokens > 256:
            print("Generation limit exceeded", flush=True)
            return None, None

    think = re.findall(r"<think>(.*?)</think>", model_res, re.DOTALL)[0].strip()
    answer = model_res[model_res.find("</think>")+len("</think>"):].strip()

    if not answer:
        return None, None
    
    return think, answer

# %%
summarize_template = \
"""<|system|>
You are a helpful assistant. You will be given a web content in markdown format. You have to provide a summary of the web content.
You have to summarize the web content inside 'summary' tag.
For example: <summary> your summarization content in markdown </summary><|end|>
<|user|>
{web_content}<|end|>
<|assistant|>
Sure! Here is the summarized version of the provided content:
<summary>"""

def web_content_summarize(web_content):
    prompt = summarize_template.format(web_content=web_content)
    # print("Question:", data['question'], flush=True)

    stream = ollama_infr(prompt=prompt, model='phi3.5:latest', temperature=0.5)
    model_res = '<summary>\n'
    n_tokens = 0

    for part in stream:
        print(part['response'], sep='', end='', flush=True)
        model_res += part['response']
        n_tokens += 1

        if n_tokens > 4000:
            break

    summary = re.findall(r"<summary>(.*?)</summary>", model_res, re.DOTALL)#[0].strip()    
    if summary:
        return summary[0].strip()
    return ''

# %%
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
    # print("URL EXTRACTED:", markdown_lines, flush=True)
    return markdown_content


def search_tool(search_str, max_results=1):
    rets = None
    with DDGS() as ddg:
        rets = list(ddg.text(keywords=search_str, region="wt-wt", max_results=7))

    str_rets = ''
    web_contents = []
    web_summary = []
    web_url = []
    n_results, i = 0, -1
    while n_results < max_results and i+1 < len(rets):
        i += 1
        # try:
        if True:
            print("Parsing url:", rets[i]['href'], flush=True)
            web_content = url_content(rets[i]['href'])[:1024*4] + " ..."
            web_content = web_content.strip()
            if web_content == '':
                continue

            # web_content = web_content_summarize(web_content=web_content)
            content = f"\n# Source {n_results+1}:"
            content += "\n" + "-" * len(content) + f"\n\n{web_content}\n\n"
            str_rets += content
            n_results += 1
        # except Exception as E:
        #     # print(E)
        #     continue
        
    return str_rets

print(search_tool('current methods used by scientists to improve predictions of climate change and its environmental impact', max_results=1))

# %%
topics = ["python math question", "git", "problem solving", "python debugging", "current knowledge", "terminal commands", "computer science"]

for _ in range(500):
    topic = random.choice(topics)
    # topic = "greeting"
    print(f"## Topic: {topic}", flush=True)
    stream = ollama_infr(question_gen_template.format(topic_name=topic), extra_stops=["</question>"])
    question = ''
    for part in stream:
        print(part['response'], sep='', end='', flush=True)
        question += part['response']
        if len(question) > 1000:
            break
    print(flush=True)

    if len(question) > 1000:
        print("Question overflow", flush=True)
        continue
    question = question.strip()

    stream = ollama_infr(search_query_template.format(query=question), extra_stops=["</search>"], temperature=0.3)
    search_str = ''
    print("Search str: ", end='', flush=True)
    for part in stream:
        print(part['response'], sep='', end='', flush=True)
        search_str += part['response']
    print(flush=True)
    search_str = search_str.strip()
    search_str = search_str.replace('"', '')

    if len(search_str) > 300:
        print("Ignoring question and search string as search_str length is greater than limit", flush=True)
        continue

    max_results = random.choice(range(3))
    context = search_tool(search_str, max_results=max_results)
    if context.strip() == '': 
        print("Empty context found", flush=True)
        continue
    
    print("### Search results:", flush=True)
    print(context, flush=True)
    print(flush=True)

    print("### Deepseek-r1", flush=True)
    think, answer = r1_response(question=question, context=context)
    if answer is None:
        print("No answer found from deepseek-r1", flush=True)
        continue

    print("\n\n### LLM As Judge", flush=True)
    stream = ollama_infr(llm_judge_prompt.format(question=question, answer=answer), extra_stops=["</rating>"])
    judge_response = ''
    for part in stream:
        print(part['response'], sep='', end='', flush=True)
        judge_response += part['response']

    rating = extract_rating(judge_response)
    print("\n\nRATING:", rating, end="", flush=True)
    print("\n--------", flush=True)

    if rating is not None and rating >= 3.0:
        write_jsonl({
            "question": question,
            "search_str": search_str,
            "search_results": context,
            "n_search_results": max_results,
            "think": think,
            "answer": answer,
            "judge_response": judge_response,
            "judge_rating": rating
        })

# %%



