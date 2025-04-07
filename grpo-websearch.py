# %%
# Reference:
# https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
# https://www.philschmid.de/mini-deepseek-r1
# https://huggingface.co/blog/open-r1/update-1
# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb#scrollTo=cXk993X6C2ZZ

#import torch._dynamo
#torch._dynamo.config.suppress_errors = True

import os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from datasets import load_dataset
import peft
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig
import transformers, gc

# from safetensors.torch import load_model, save_model

import random
import re
import sys
import json
import ast
from copy import deepcopy

from webtool.webtool import webtool_def, tool_call_extract
from datagen.prompt_templates import question_gen_template_phi3, grpo_judge_prompt_phi3
from datagen.main import ollama_infr, topics


MODEL_PATH = "quwsarohi/SmolThink"
ITER_STEP = 10_000

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="mps",
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_cache=False,
    tie_word_embeddings=True,
)

# Gradient checkpointing - Could take more memory in MPS
model.gradient_checkpointing_enable(dict(use_reentrant=False))
print(f"Model took {model.get_memory_footprint()/1e9:.2f} GB of space (with buffer)")
# print(model)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    add_bos_token=True,
    add_eos_token=True,
)

tokenizer.pad_token = tokenizer.unk_token
streamer = TextStreamer(tokenizer, skip_prompt=False)

# %%

class DatasetGen_v1(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dlen):
        self.tokenizer = tokenizer
        self.dlen = dlen
        self.cache = {}

    def __len__(self):
        return self.dlen

    def get_question(self):
        topic = random.choice(topics)
        stream = ollama_infr(
            question_gen_template_phi3.format(topic_name=topic), 
            model='phi3.5', 
            extra_stops=["</question>"], 
            temperature=0.99,
            top_k=150,
            num_predict=64
        )
        question = ''
        for part in stream:
            # print(part['response'], sep='', end='', flush=True)
            question += part['response']
            if len(question) > 1000:
                return None
        # print(flush=True)
        return question.strip()

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        question = None
        while not question:
            question = self.get_question()

        prompt = self.tokenizer.apply_chat_template(
            [{"role":"user", "content": question}], 
            tools=[webtool_def], 
            tokenize=False
        )
        # print(prompt, flush=True)
        # return self.tokenizer(prompt)
        self.cache[idx] = {'prompt': prompt, }#'question': question}
        return self.cache[idx]

dataset = DatasetGen_v1(tokenizer, ITER_STEP)
# print(dataset[0])
# sys.exit()

def extract_rating(input_str:str):
    data = input_str.split('<rating>')[1]
    data = data.strip().split()
    for d in data:
        try:    return float(d)
        except: continue
    return 0


def judge_search_str(question:str, search_str:str, verbose=False):
    prompt = grpo_judge_prompt_phi3.format(user_question=question, search_string=search_str)
    stream = ollama_infr(
        prompt, 
        model='phi3.5',
        temperature=0.5,
        num_predict=256,
        extra_stops=["</rating>"]
    )
    judge_response = ''
    for part in stream:
        if verbose: print(part['response'], sep='', end='', flush=True)
        judge_response += part['response']
    if verbose: print(flush=True)
    try:
        rating = extract_rating(judge_response)
        return max(rating - 1, 0)
    except:
        return 0


def tool_call_score(prompts, completions, **kwargs):
    # Compile regex to capture content inside <tool_call> tags, allowing for whitespace/newlines.
    # pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    # matches = [pattern.findall(text) for text in completions]
    # print(prompts[0])
    scores = []
    for i, prompt in enumerate(prompts):
        try:
            score = 0.0
            tool_call = tool_call_extract(completions[i])
            if tool_call is None:
                scores.append(score)
                continue
            elif tool_call.get('name', None) == 'web_search':
                score += 0.5
            search_str = tool_call.get('arguments', {}).get('search_str', '')
            # score += max(3.0, len(search_str) * 0.025)
            if search_str:
                tag = '<|im_start|>user'
                pos = prompt.find(tag) + len(tag)
                question = prompt[pos:].replace('<|im_end|>', '')
                score += judge_search_str(question=question, search_str=search_str)
            scores.append(score)
        except Exception as E:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(E, "|", "line:", exc_tb.tb_lineno, flush=True)
            scores.append(score)

    print("---- QUESTION ----", flush=True)
    print(prompts[0], flush=True)
    print("---- COMPLETION ----", flush=True)
    print(completions[0], flush=True)
    print("--------", flush=True)
    print("Tool-call Score:", scores)
    return scores


class MpsCacheClearCallback(transformers.TrainerCallback):
    def __clearmem(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        # Note: Clearing model gradients
        # for param in model.parameters():
            # param.grad = None
        # print("\nMEMORY CLEARED\n")
    #def on_step_begin(self, *args, **kwargs):      self.__clearmem()
    def on_step_end(self, *args, **kwargs):        self.__clearmem()
    #def on_substep_end(self, *args, **kwargs):     self.__clearmem()
    #def on_evaluate(self, *args, **kwargs):        self.__clearmem()
    #def on_optimizer_step(self, *args, **kwargs):  self.__clearmem()
    #def on_predict(self, *args, **kwargs):         self.__clearmem()
    #def on_prediction_step(self, *args, **kwargs): self.__clearmem()
gc.collect()


training_args = GRPOConfig(
    use_vllm = False,
    learning_rate = 5e-5,
    # adam_beta1 = 0.9,
    # adam_beta2 = 0.99,
    weight_decay = 0.2,
    warmup_ratio = 50 / len(dataset),
    logging_steps=5,
    max_steps=len(dataset),
    save_steps = 10,
    save_total_limit=1,
    ds3_gather_for_generation=False,
    lr_scheduler_type = "cosine",
    # Memory reduction
    optim = "adafactor",    # adamw_torch
    # Memory reduction
    bf16 = True,
    bf16_full_eval=True,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    # Memory reduction
    torch_empty_cache_steps=1,
    num_generations = 2, # Decrease if out of memory
    max_prompt_length = 512,
    max_completion_length = 128,
    temperature=0.9,
    # top_k=15,   # default is 50
    # repetition_penalty = 1.1, # default is 1
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = f"/Users/ohi/Documents/GitHub/PersonalAssistant/weights/SmolThink-grpo",
    # Memory reduction
    dataloader_pin_memory=False,
    # Gradient checkpointing - could take more memory in MPS
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # torch_compile=True,
    # include_tokens_per_second=True
    disable_tqdm=False
)

model.config.use_cache = False
model.generation_config.do_sample = True
model.generation_config.temperature = 0.9
model.generation_config.top_k = 20
# model.generation_config.eos_token_id = tokenizer.eos_token_id

trainer = GRPOTrainer(
    model=model,
    # processing_class = tokenizer,
    reward_funcs = [
        # correctness_reward_func,
        tool_call_score
    ],
    args=training_args,
    train_dataset=dataset,
    # callbacks=[MpsCacheClearCallback()]
    # peft_config=peft_config, #get_peft_config(model_config),
)

# try:
#     trainer.train(resume_from_checkpoint=True)
# except ValueError as E:
#     print("No checkpoint found")
#     trainer.train(resume_from_checkpoint=False)

trainer.train(resume_from_checkpoint=False)

# %%



