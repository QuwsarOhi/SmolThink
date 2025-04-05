
# Reference:
# * https://huggingface.co/agents-course/notebooks/blob/main/bonus-unit1/bonus-unit1.ipynb
# * https://colab.research.google.com/#scrollTo=29da85c8-33bf-4864-aed7-733cbe703512&fileId=https%3A//huggingface.co/agents-course/notebooks/blob/main/bonus-unit1/bonus-unit1.ipynb


# PEFT implementation
# https://colab.research.google.com/drive/12pMorxvLV-VwjuNBM76L4xXnzVYg57iB?usp=sharing#scrollTo=Gzke5ccani9m

# Huggingface Inference
# https://huggingface.co/docs/transformers/v4.49.0/llm_optims?static-kv=advanced+usage%3A+control+Static+Cache#fine-tuning-with-torchcompile-and-padding-free-data-collation

# Inference Cache
# https://huggingface.co/docs/transformers/en/kv_cache

# Ollama
# https://github.com/ollama/ollama/issues/1754
# https://github.com/ollama/ollama/blob/main/docs/import.md

# LoRA vs Full FT
# * https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2

import os
import json
from copy import deepcopy
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, StoppingCriteria

from webtool.webtool import search_tool, webtool_def

import warnings
warnings.filterwarnings("ignore")

# os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def get_latest_checkpoint(base_directory):
    checkpoint_dirs = []
    
    # List all directories in the base directory
    for dir_name in os.listdir(base_directory):
        if re.match(r'checkpoint-\d+', dir_name):  # Match pattern "checkpoint-N"
            checkpoint_dirs.append(dir_name)
    
    if not checkpoint_dirs:
        return None  # No checkpoints found
    
    # Sort directories based on numerical value
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
    
    return os.path.join(base_directory, latest_checkpoint)

SIZE = "360M" #"135M" #"360M"
# MODEL_PATH = f"HuggingFaceTB/SmolLM2-{SIZE}-Instruct"
# MODEL_PATH = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
# SAVE_PATH = "SmolThink-Qwen-sft"

FILE_PATH = get_latest_checkpoint(f"/Users/ohi/Documents/GitHub/PersonalAssistant/weights/SmolThink-{SIZE}-sft-websearch")
# FILE_PATH = get_latest_checkpoint(f"/Users/ohi/Documents/GitHub/PersonalAssistant/weights/SmolLM2-{SIZE}-grpo")

TOKENIZER_PATH = FILE_PATH
LORA_PATH = os.path.join(FILE_PATH, "smolthink")
print("LoRA checkpoint:", LORA_PATH)

# config = peft.PeftConfig.from_pretrained(LORA_PATH)
model = AutoModelForCausalLM.from_pretrained(
    FILE_PATH,
    # "/Users/ohi/Documents/GitHub/PersonalAssistant/weights/SmolThink-360M-sft/checkpoint-28800",
    # MODEL_PATH,
    device_map="mps",
    low_cpu_mem_usage=True,
    attn_implementation='eager', # 'sdpa'
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_cache=False,
    tie_word_embeddings=True,
)

print("model.config.tie_word_embeddings:", model.config.tie_word_embeddings)
print(f"Model took {model.get_memory_footprint()/1e9:.2f} GB of space (with buffer)")
print(sum(p.numel() for p in model.parameters()) / 1e6)

# model = peft.PeftModel.from_pretrained(
#    model,
#    LORA_PATH,
#    is_trainable=False, # 👈 here,
# )

# lora_param = 0
# lora_layers = 0
# for name, param in model.named_parameters():
#     if 'lora' in name:
#         lora_param += param.numel()
#         lora_layers += 1

# def into_million(val):
#     return f"{val / 1000 / 1000 :.2f} million"

# print(f"Total LoRA params: {into_million(lora_param)}   |   Total LoRA layers: {lora_layers}")

# model = model.merge_and_unload(safe_merge=True).eval().to(torch.bfloat16)
print(f"Merged model+LoRA took {model.get_memory_footprint()/1e9:.2f} GB of space (with buffer)")
print(sum(p.numel() for p in model.parameters()) / 1e6)

# print(model)
# print(model.base_model.embed_tokens.weight)
# print(model.lm_head.weight)
# print("Are LoRA weight of embed_tokens and lm_head same?", torch.equal(model.base_model.model.model.embed_tokens.modules_to_save["smolthink"].weight, model.base_model.model.lm_head.modules_to_save["smolthink"].weight))
# model.base_model.model.model.embed_tokens.modules_to_save["smolthink"].weight = model.base_model.model.lm_head.modules_to_save["smolthink"].weight
# print("LoRA embed_tokens and lm_head sharing the same memory?", model.base_model.model.model.embed_tokens.modules_to_save["smolthink"].weight.data.data_ptr() == model.base_model.model.lm_head.modules_to_save["smolthink"].weight.data.data_ptr())
print("Is model embed_tokens and lm_head sharing same memory?", model.base_model.embed_tokens.weight.data.data_ptr() == model.lm_head.weight.data.data_ptr())
print("Do model embed_tokens and lm_head share same weight?", torch.equal(model.base_model.embed_tokens.weight, model.lm_head.weight))

# model.base_model.embed_tokens.weight = model.lm_head.weight
# print("COPEID")
# print("Is model embed_tokens and lm_head sharing same memory?", model.base_model.embed_tokens.weight.data.data_ptr() == model.lm_head.weight.data.data_ptr())
# print("Do model embed_tokens and lm_head share same weight?", torch.equal(model.base_model.embed_tokens.weight, model.lm_head.weight))



chat_template = """{%- if tools %}
    {{- '<|endoftext|><|im_start|>system\\n' }}
        {%- if messages[0]['role'] == 'system' %}
            {- messages[0]['content'] }}
        {%- else %}
            {{- 'You are a helpful AI assistant named SmolThink.' }}
        {%- endif %}
    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> tags:\\n<tools>\" }}
    {%- for tool in tools %}
        {{- \"\\n\" }}
            {{- tool | tojson }}
    {%- endfor %}
    {{- \"\\n</tools>\\n\\nYou first think/plan inside <think></think> tags.\\nThen for each function call, return a json object with function name and arguments within <tool_call></tool_call> tags.<|im_end|>\\n\" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|endoftext|><|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|endoftext|><|im_start|>system\\nYou are a helpful AI assistant named SmolThink. First plan/reason/code/validate inside <think></think> tag and provide final answer to user query inside <answer></answer> tag.<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == \"assistant\" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\\n<tool_call>\\n{\"name\": \"' }}
            {{- tool_call.name }}
            {{- '\", \"arguments\": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == \"tool\" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- message.content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n<think>\\n' }}
{%- endif %}"""

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH,
    add_bos_token=True,
    add_eos_token=True,
)
tokenizer.chat_template = chat_template
streamer = TextStreamer(tokenizer, skip_prompt=False)

print("\n-----\n")



# Ref: https://github.com/nestordemeure/stop_word/blob/main/stop_word_criteria.py

from transformers import StoppingCriteria
from typing import List

class StopWordCriteria(StoppingCriteria):
    """
    A stopping criteria that halts the text generation process if any specified stop word is encountered.
    
    Inspired by https://discuss.huggingface.co/t/implimentation-of-stopping-criteria-list/20040/9
    And: https://github.com/outlines-dev/outlines/blob/main/outlines/generate/api.py
    """
    
    def __init__(self, tokenizer: AutoTokenizer, prompts: List[str], stop_words: List[str] = [], check_every: int = 1):
        """
        Initializes the StopWordCriteria with the necessary parameters for checking stop words during text generation.
        
        Parameters:
            tokenizer (AutoTokenizer): The tokenizer for encoding prompts and stop words.
            prompts (List[str]): Initial prompts used for generation, needed to determine where generated text begins.
            stop_words (List[str]): Words that trigger the stopping of generation when detected.
            check_every (int): Frequency of checking for stop words in the token stream (a performance optimization, use 1 to cut it out).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.input_sizes = [self.tokenizer.encode(prompt, return_tensors="pt").size(-1) for prompt in prompts]
        self.stop_words = stop_words
        self.max_stop_word_size = max((self.tokenizer.encode(word, return_tensors="pt").size(-1) for word in stop_words), default=0)
        self.check_every = check_every

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Determines whether to stop generation based on the presence of stop words.
        
        Stops if a stop word is found in *all* batch elements *and* the sequence length is a multiple of `check_every`.
        Note: Delay in stopping may occur if `check_every > 1`.

        Parameters:
            input_ids (torch.LongTensor): Generated token IDs.
            scores (torch.FloatTensor): Generation scores for each token. Not used here.
        
        Returns:
            bool: True to stop generation, False to continue.
        """
        batch_size, seq_len = input_ids.shape
        
        # Skip check if no stop words are defined or it is not yet time to check
        if (len(self.stop_words) == 0) or (seq_len % self.check_every != 0):
            return False
        
        for i in range(batch_size):
            # Calculate starting index for new tokens
            prompt_size = self.input_sizes[i]
            max_new_tokens = (2 * self.max_stop_word_size) + self.check_every
            latest_tokens = input_ids[i, prompt_size:][-max_new_tokens:]
            
            # Check for stop words in the decoded text
            if not any(word in self.tokenizer.decode(latest_tokens, skip_special_tokens=True) for word in self.stop_words):
                return False  # Continue generation if any batch item lacks stop words
                
        return True  # Stop generation if all conditions are met

    def extract_answers(self, input_ids: torch.LongTensor, strip_stopword: bool = True) -> List[str]:
        """
        Extracts generated answers by removing prompts and optionally stopping at the first stop word.
        
        Parameters:
            input_ids (torch.LongTensor): Generated token IDs.
            strip_stopword (bool): Determines whether the stop word is removed from the output.
            
        Returns:
            List[str]: Extracted answers, with or without stop words.
        """
        batch_size, _ = input_ids.shape
        result = []
        
        for i in range(batch_size):
            # Decode generated tokens to text, excluding the prompt
            prompt_size = self.input_sizes[i]
            answer_tokens = input_ids[i, prompt_size:]
            answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            # Find the first occurrence of any stop word
            lower_stop_index = len(answer_text)  # Default to end of text
            for word in self.stop_words:
                stop_index = answer_text.find(word)
                if stop_index != -1:
                    # Adjust stop index based on whether we're stripping the stop word
                    stop_index += 0 if strip_stopword else len(word)
                    lower_stop_index = min(stop_index, lower_stop_index)
            
            # Cut the text at the first stop word found (if any)
            answer_text = answer_text[:lower_stop_index]
            result.append(answer_text)
        
        return result

# https://huggingface.co/docs/transformers/main/en/kv_cache#offloaded-static-cache
# model.generation_config.cache_implementation = "static"
# model.generate = torch.compile(model.generate, mode="reduce-overhead", fullgraph=True)

from torch.nn.attention import SDPBackend, sdpa_kernel

def inference(input_text, max_new_tokens=100, stop_words=[], **kwargs):
    input_ids = tokenizer(input_text, return_tensors="pt")

    stopping_criteria = StopWordCriteria(
        tokenizer=tokenizer, 
        prompts=[input_text], 
        stop_words=stop_words
    )

    with torch.inference_mode():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            outputs = model.generate(
                streamer=streamer,
                # **input_ids
                max_new_tokens=max_new_tokens,
                # use_cache=False,
                **kwargs,
                # stop_strings=["</answer>"],
                input_ids=input_ids['input_ids'].to('mps'),
                attention_mask=input_ids['attention_mask'].to('mps'),
                stopping_criteria=[stopping_criteria]
            )
    # Only generate output
    input_token_len = input_ids['input_ids'].shape[-1]
    print("Total number of tokens:", len(outputs[0][input_token_len:]))
    return tokenizer.decode(outputs[0][input_token_len:], skip_special_tokens=False)



def tool_parse(tool_call:str):
    ret = None
    try:
        ret = literal_eval(tool_call)
    except Exception:
        pass

    _tool_call = tool_call.replace("'", '"')
    ret = json.loads(_tool_call)
    return ret

def tool_call_extract(inp_str:str):
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    tool_calls = pattern.findall(inp_str)
    if tool_calls:
        tool_call = tool_parse(tool_calls[0])
        return tool_call
    return None

def remove_think(inp_str:str):
    inp_str = deepcopy(inp_str)
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    thinks = pattern.findall(inp_str)
    for think in thinks:
        inp_str = inp_str.replace(think, '')
    return inp_str



while True:
    user_message = input("Your input: ")
    base_prompt = tokenizer.apply_chat_template([
        {"role": "user", "content": user_message}
    ], tools=[webtool_def], tokenize=False, add_generation_prompt=True) #+ "Let's do an information_search to search relevant information before answering user question\n</think>\n<tool_call>\n{'name': 'information_search', 'arguments': {'search_str': '"

    gen = inference(
        base_prompt, 
        low_memory=True,
        do_sample=False,
        # temperature=0.4,
        repetition_penalty=1.1,
        max_new_tokens=512*2,
        stop_words=["</tool_call>"]
    )

    tool_call = tool_call_extract(gen)
    # print(tool_call)
    # break
    # what is ci/cd?
    try:
        result, urls = search_tool(tool_call['arguments']['search_str'], trim=1024*6)#[:1024*2] + "..."
        result += f"\n\n\nUser question: {user_message}\n"
    except Exception as E:
        print("Cannot process tool_call. Falling back", flush=True)
        continue
    
    print("-------")
    base_prompt = tokenizer.apply_chat_template([
        {"role": "user", "content": user_message},
        {"role": "tool", "content": result}
    ], tools=None, tokenize=False, add_generation_prompt=True) #+ "</think>\n<answer>\n"

    gen = inference(
        base_prompt, 
        low_memory=True,
        do_sample=False,
        # temperature=0.4,
        repetition_penalty=1.1,
        max_new_tokens=1024,
    )

    print("-"*50, end='\n\n')


    # sys.exit()