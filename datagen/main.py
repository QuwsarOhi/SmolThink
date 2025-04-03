import random
import json
import re

import ollama
from prompt_templates import *
from webtool.webtool import search_tool


# https://github.com/ollama/ollama-python/blob/main/examples/async-chat-stream/main.py
def ollama_infr(prompt, extra_stops=[], model='deepseek-r1:7b', temperature=0.7):
    # https://github.com/ollama/ollama-python/blob/00eafed0faa5dea6879a8eb3229c7a8f2439abb4/ollama/_types.py#L93
    return ollama.generate(
        model = model,
        # system = system,
        # Raw is set to true to feed the question as needed
        raw=True,
        prompt = prompt,
        stream = True,
        # Number of seconds to keep the connection alive
        keep_alive=-1, # Will keep the model loaded,
        options = {
            'stop': [
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
            ] + extra_stops,
            'temperature': temperature,
            # 'top_k': 1,
            'cache': True,
            # 'tfs_z': 2.0,
            'num_ctx': 6000,
            # 'temperature': 0.0,
            # 'top_p': 0.0
        },
    )

DATA_PATH = "search_data.jsonl"
def write_jsonl(data: dict):
    with open(DATA_PATH, "a") as f:
        f.write(json.dumps(data) + "\n")


def extract_rating(input_str:str):
    data = input_str.split('<rating>')[1]
    data = data.strip().split()
    for d in data:
        try:    return float(d)
        except: continue
    return None


def r1_response(question, context):  
    model_res = '<think>\n'
    if context:
        prompt = deepseek_r1_context.format(question=question, context=context)
    else:
        model_res += "</think>"
        prompt = deepseek_r1_nocontext.format(question=question)
        
    stream = ollama_infr(prompt=prompt, model='deepseek-r1:7b', temperature=0.5)
    n_think_tokens = 0
    think_finished = False

    for part in stream:
        print(part['response'], sep='', end='', flush=True)
        model_res += part['response']

        if not think_finished and '</think>' in model_res:
            think_finished = True
        if not think_finished:
            n_think_tokens += 1

        if n_think_tokens > 386:
            print("Generation limit exceeded", flush=True)
            return None, None
        
        if len(model_res) > 6000:
            return None, None

    think = re.findall(r"<think>(.*?)</think>", model_res, re.DOTALL)[0].strip()
    answer = model_res[model_res.find("</think>")+len("</think>"):].strip()

    if not answer:
        return None, None
    
    return think, answer



topics = [
    # "casual greeting"
    "python math question", 
    "git",
    "docker",
    "coding problem solving", 
    "software development", 
    "general knowledge", 
    "terminal commands", 
    "math questions",
    "computer science",
    "algorithms (code in python)"
]

while True:
    topic = random.choice(topics)
    # topic = "greeting"
    print(f"\n\n# Topic: {topic}", flush=True)
    if 'casual' in topic:
        stream = ollama_infr(casual_conv_template, extra_stops=["</question>"], temperature=0.9)
    else:
        stream = ollama_infr(question_gen_template.format(topic_name=topic), extra_stops=["</question>"], temperature=0.9)
    question = ''
    for part in stream:
        print(part['response'], sep='', end='', flush=True)
        question += part['response']
        if len(question) > 1000:
            break
    print(flush=True)

    question = question.strip()
    if len(question) > 1000:
        print("Question char overflow", flush=True)
        continue
    
    context, source_urls, search_str = '', '', ''
    if 'casual' not in topic:
        stream = ollama_infr(search_query_template.format(query=question), extra_stops=["</search>"], temperature=0.3)
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

        max_results = random.choice(range(2, 4))
        context, source_urls = search_tool(search_str, max_results=max_results)
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

    try:
        rating = extract_rating(judge_response)
    except:
        continue
    
    print("\n\nRATING:", rating, end="", flush=True)
    print("\n--------", flush=True)

    if rating is not None and rating >= 3.0:
        write_jsonl({
            "question": question,
            "search_str": search_str,
            "search_results": context,
            "source_urls": source_urls,
            "think": think,
            "answer": answer,
            "judge_response": judge_response,
            "judge_rating": rating
        })





