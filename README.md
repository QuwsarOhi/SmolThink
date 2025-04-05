# SmolThink: A Small Model That Tries to Think

**SmolThink** model is a Continued Supervised Fine-Tuned version of [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) on **Deepseek-R1** distilled dataset.

## Trainin Process

The model was trained on a mixture of small Chain of Thoughts (CoT) and some long CoT dataset. Small CoT dataset mixture was used as the model is small and it is was reported that small models struggle to produce long reasoning chain [ref](https://arxiv.org/abs/2502.12143).

The SFT dataset had been created from the following data mixtures:

* [ServiceNow-AI/R1-Distill-SFT](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT)
* [Jofthomas/hermes-function-calling-thinking-V1](https://huggingface.co/datasets/Jofthomas/hermes-function-calling-thinking-V1)
* [GeneralReasoning/GeneralThought-195K](https://huggingface.co/datasets/GeneralReasoning/GeneralThought-195K)
* [open-r1/codeforces-cots](https://huggingface.co/datasets/open-r1/codeforces-cots)
* [XeTute/Open-Coding-Thoughts (currently unavailable)](https://huggingface.co/datasets/XeTute/Open-Coding-Thoughts)
* Custom tool calling and websearch summarizing dataset generated using **Phi-3.5**, **Qwen2: deepseek-r1:7b**. Data generation pipeline is described [here](#)

The datasets were filtered by removing the contents having CoT length more than 256 words. The model was trained to produce tool calls. As being a small language model, the model does not memorize certain things (example: how to bake cake). Rather, if used with web search, the model may produce good quality of answer regardless of the size.

The model was supervised fine-tuned in two phases:
* The model was initially trained with a rolling context length of `832` with a token stride of `832/8`. Dataset snapshot is available as [merged_dataset_phase1](#).
* The model was again trained with a rolling context length of `3072` with a token stride of `768`. Dataset snapshot is available as [merged_dataset_phase2](#)

The model is still under training and the whole training and dataset mixtures would be published soon. The model is trained on MacBook Air with 16GB unified memory.

## Usage

### General Usage

Use the following code to load and use the model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = 'mps'

tokenizer = AutoTokenizer.from_pretrained(
    "quwsarohi/SmolThink"
)

model = AutoModelForCausalLM.from_pretrained(
    "quwsarohi/SmolThink",
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_cache=False,
    tie_word_embeddings=True,
).to(device)

messages = [{"role": "user", "content": "What is the capital of France."}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
print(input_text)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
print(tokenizer.decode(outputs[0]))
```

### WebSearch Tool Integration

The model is further trained to do web search using a special websearch tool. The following code could be used to use the web searching capability.

```python
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

base_prompt = tokenizer.apply_chat_template([
    {"role": "user", "content": "What is the current stock price of Apple?"}
], tools=[webtool_def], tokenize=False, add_generation_prompt=True)
print(base_prompt)

inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
print(tokenizer.decode(outputs[0]))
```

For live websearch integrated chat, execute `python3 chat.py` to chat with SmolThink with websearch integration. The implementation uses **Dolcing** and **DuckDuckGo** free search API.
