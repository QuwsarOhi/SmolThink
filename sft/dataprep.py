import ast
import json
import os

# from safetensors.torch import load_model, save_model
import random
import re
from copy import deepcopy
from datasets import Dataset, concatenate_datasets, load_dataset


def r1_distill(reasoning_len, n_data=None):
    def length_filter(data, limit):
        return 0 < data["thought_len"] <= limit and 0 < data["answer_len"]

    def data_process(data):
        thought_len, answer_len = 0, 0
        for idx, conv in enumerate(data["reannotated_messages"]):
            # print(conv)
            role = conv["role"]
            if role == "assistant":
                reply = data["reannotated_messages"][idx]["content"]
                # print(reply)
                thought = re.findall(r"<think>(.*?)</think>", reply, re.DOTALL)
                thought = "".join(thought).strip()
                thought_len += len(thought.split())  # len(tokenizer.encode(thought))

                end_tag = "</think>"
                if end_tag in reply:
                    answer = reply[reply.find(end_tag) + len(end_tag) :]
                    answer = answer.strip()
                else:
                    answer = ""
                if thought.lower() == answer.lower():
                    answer = ""
                answer_len += len(answer.split())  # len(tokenizer.encode(answer))
                data["reannotated_messages"][idx]["content"] = (
                    f"<think>\n{thought}\n</think>\n<answer>\n{answer}\n</answer>"
                )

        if "system" in data:
            del data["system"]
        data["thought_len"] = thought_len
        data["answer_len"] = answer_len
        data["source"] = "ServiceNow-AI/R1-Distill-SFT"
        return data

    r1_dataset = load_dataset("ServiceNow-AI/R1-Distill-SFT", "v1")["train"]
    r1_dataset.shuffle(123)
    delete_keys = list(r1_dataset.column_names)
    r1_dataset = r1_dataset.map(data_process)
    r1_dataset = r1_dataset.filter(lambda x: length_filter(x, reasoning_len))
    r1_dataset = r1_dataset.map(
        lambda x: {
            "conversations": tokenizer.apply_chat_template(
                x["reannotated_messages"], tools=None, tokenize=False
            )
        }
    )
    r1_dataset = r1_dataset.remove_columns(delete_keys)
    if n_data:
        r1_dataset = r1_dataset.select(range(n_data))
    print("R1-distill dataset length (after filter):", len(r1_dataset))
    return r1_dataset


def tool_shuffle(tool):
    if not isinstance(tool, list):
        tool = [tool]
    if tool:
        assert not isinstance(tool[0], str), f"Tool type should not be a str: type-{type(tool[0])}"
    random.shuffle(tool)
    reps = [str(tool)]
    for c in [None, 1, 2, 3, 4]:
        reps.append(json.dumps(tool, indent=c))
    return random.choice(reps)


def extract_tag(input_str, tag):
    tool_def = re.findall(f"<{tag}>(.*?)</{tag}>", input_str, re.DOTALL)
    tool_def = map(str.strip, tool_def)
    tool_def = filter(lambda x: len(x) > 0, tool_def)
    return list(tool_def)


def hermes_fc_thinking(raw_data):
    data = deepcopy(raw_data["conversations"])
    seq = []
    tool_def = None
    tool_names = None
    for d in data:
        if d["role"] == "system":
            tool_def = extract_tag(d["content"], "tools")
            if len(tool_def) != 0:
                try:
                    tool_def = ast.literal_eval(tool_def[0])
                    tool_names = [tool["function"]["name"] for tool in tool_def]
                    seq.append({
                        "role": "system", 
                        "content": TOOL_TEMPLATE.format(tools=tool_shuffle(tool_def))
                    })
                    continue
                except: 
                    return {"conversations": "", "source": "Jofthomas/hermes-function-calling-thinking-V1"}
            else:
                return {"conversations": "", "source": "Jofthomas/hermes-function-calling-thinking-V1"}

        seq.append({})
        seq[-1]["role"] = {
            "human": "user",
            "model": "assistant",
            "system": "system",
            "tool": "tool",
        }[d["role"]]
        seq[-1]["content"] = d["content"]

        if seq[-1]["role"] == "assistant":
            seq[-1]["content"] = seq[-1]["content"].replace("<think>", "<think>\n")
            seq[-1]["content"] = seq[-1]["content"].replace("</think>", "</think>\n")

            tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", seq[-1]["content"], re.DOTALL)
            sanitized_tool_calls = []

            if tool_calls:
                for tool_call in tool_calls:
                    try:
                        tool_call = ast.literal_eval(tool_call.strip())
                        if tool_call["name"] not in tool_names:
                            raise NotImplementedError
                        sanitized_tool_calls.append(tool_call)
                    except:
                        return {"conversations": "", "source": ""}
                
                sanitized_tool_calls = json.dumps(sanitized_tool_calls, indent=None)
                seq[-1]["content"] = f"<tool_call>{sanitized_tool_calls}</tool_call>"
            else:
                seq[-1]["content"] = (f"<think>\n</think>\n<answer>\n{seq[-1]['content']}\n</answer>")
        
        if seq[-1]["role"] == "tool":
            seq[-1]["content"] = seq[-1]["content"].replace("<tool_response>", "")
            seq[-1]["content"] = seq[-1]["content"].replace("</tool_response>", "")
            seq[-1]["content"] = json.dumps(ast.literal_eval(seq[-1]["content"].strip()))
        # seq[-1]['content'] = d['value']

    # random.shuffle(tool_def)
    ret = tokenizer.apply_chat_template(
        seq, 
        # tools=tool_def,
        tokenize=False, 
        add_generation_prompt=False,
    )
    return {"conversations": ret, "source": "Jofthomas/hermes-function-calling-thinking-V1"}

fc_dataset = load_dataset("Jofthomas/hermes-function-calling-thinking-V1")["train"]
fc_dataset = fc_dataset.map(hermes_fc_thinking)
fc_dataset = fc_dataset.filter(lambda x: len(x["conversations"]) > 0)
# fc_dataset = fc_dataset.select(range(150))
print("Function calling dataset length (after filter):", len(fc_dataset))


def tool_call_process(data):
    new_data = {
        'prompt': '',
        'valid': False,
        'tool': '',
        'tool_call': '',
        'source': 'HuggingFaceTB/smoltalk/apigen-80k'
    }
    tool_def = None
    pattern = re.compile(r'<tools>(.*?)</tools>', re.DOTALL)
    tool_def = pattern.match(data['messages'][0]['content'])

    try:
        content = data['messages'][0]['content']
        tool_def = re.findall(r"<tools>(.*?)</tools>", content, re.DOTALL)[0]
        tool_def = json.loads(tool_def)
        new_data['tool'] = json.dumps(tool_def)
    except:
        return new_data

    seq = [{"role": "system", "content": SYS_TEMPLATE.format(tools=tool_shuffle(tool_def))}]
    
    for s in data['messages']:
        if s['role'] == 'system': continue
        if s['role'] == 'user':
            seq.append(s)
        elif s['role'] == 'assistant':
            tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", s['content'], re.DOTALL)
            if tool_calls:
                tool_calls = json.loads(tool_calls[0])
                tool_calls = json.dumps(tool_calls, indent=None) #random.choice([0, 2, 4]))
                new_data['tool_call'] = tool_calls
                seq.append({"role": "assistant", "content": f"<think>\n</think>\n<tool_call>{tool_calls}</tool_call>"})   
            else:
                s['content'] = f"<think>\n</think>\n<answer>\n{s['content']}\n</answer>"
                seq.append(s)
        else:
            break
    
    # new_data['valid'] = True
    tool_def = [{"type": "function", "function": e} for e in tool_def]
    new_data['conversations'] = tokenizer.apply_chat_template(seq, tokenize=False, add_generation_prompt=False)
    return new_data

smoltalk_fc_dataset = load_dataset("HuggingFaceTB/smoltalk", "apigen-80k")['train'] #['train']#.select(range(30))#.select(range(2))
smoltalk_fc_dataset = smoltalk_fc_dataset.map(tool_call_process)
# smoltalk_fc_dataset = smoltalk_fc_dataset.remove_columns(smoltalk_fc_dataset.column_names)
# smoltalk_fc_dataset = smoltalk_fc_dataset.filter(lambda x: x['valid'] == True)
print("Smoltalk function calling dataset length (after filter):", len(smoltalk_fc_dataset))


def data_process(data, source):
    new_data = {}
    seq = []
    
    for s in data['messages']:
        if s['role'] == 'system': continue
        if s['role'] == 'user':
            seq.append(s)
        elif s['role'] == 'assistant':
                s['content'] = f"<think>\n</think>\n<answer>\n{s['content']}\n</answer>"
                seq.append(s)
        else:
            raise NotImplementedError(f"Role: {s['role']} not recognized")
    
    new_data['conversations'] = tokenizer.apply_chat_template(seq, tokenize=False, add_generation_prompt=False)
    new_data['source'] = source
    return new_data

hf_everyday_conv = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations")
hf_everyday_conv = concatenate_datasets([hf_everyday_conv['train'], hf_everyday_conv['test']])
col_names = hf_everyday_conv.column_names
hf_everyday_conv = hf_everyday_conv.map(data_process, fn_kwargs={"source": "HuggingFaceTB/smoltalk/everyday-conversations"})
hf_everyday_conv = hf_everyday_conv.remove_columns(col_names)
print("Smoltalk everyday conv dataset length (after filter):", len(hf_everyday_conv))

hf_constraints = load_dataset("HuggingFaceTB/smoltalk", "smol-constraints")
hf_constraints = concatenate_datasets([hf_constraints['train'], hf_constraints['test']])
col_names = hf_constraints.column_names
hf_constraints = hf_constraints.map(data_process, fn_kwargs={"source": "HuggingFaceTB/smoltalk/smol-constraints"})
hf_constraints = hf_constraints.remove_columns(col_names)
print("Smoltalk constraint dataset length (after filter):", len(hf_constraints))

hf_oss = load_dataset("HuggingFaceTB/smoltalk", "self-oss-instruct")
hf_oss = concatenate_datasets([hf_oss['train'], hf_oss['test']])
col_names = hf_oss.column_names
hf_oss = hf_oss.map(data_process, fn_kwargs={"source": "HuggingFaceTB/smoltalk/self-oss-instruct"})
hf_oss = hf_oss.remove_columns(col_names)
print("Smoltalk oss/python dataset length (after filter):", len(hf_oss))


def data_process(data):
    new_data = {}
    seq = []
    
    for s in data['messages']:
        if s['role'] in ['user', 'system']:
            seq.append(s)
        elif s['role'] == 'assistant':
                s['content'] = f"<think>\n</think>\n<answer>\n{s['content']}\n</answer>"
                seq.append(s)
        else:
            raise NotImplementedError(f"Role: {s['role']} not recognized")
    
    new_data['conversations'] = tokenizer.apply_chat_template(seq, tokenize=False, add_generation_prompt=False)
    new_data['source'] = "HuggingFaceTB/smoltalk/everyday-conversations"
    return new_data

hf_systemchats = load_dataset("HuggingFaceTB/smoltalk", "systemchats-30k")
hf_systemchats = concatenate_datasets([hf_systemchats['train'], hf_systemchats['test']])
col_names = hf_systemchats.column_names
hf_systemchats = hf_systemchats.map(data_process)
hf_systemchats = hf_systemchats.remove_columns(col_names)
print("Smoltalk system chat dataset length (after filter):", len(hf_systemchats))


def general_thought(reasoning_len, n_data=None):
    def generalreason_conv(data):
        history = None
        data["empty"] = "true"
        if "prev_message" in data:
            history = data["prev_message"]
        if not history:
            history = []

        if history and history[0]["role"] == "system":
            del history[0]

        for idx, h in history:
            if history[idx]["role"] == "assistant":
                history[idx]["content"] = (
                    f"<think>\n</think>\n<answer>\n{history[idx]['content']}\n</answer>"
                )

        if data["model_reasoning"]:
            data["empty"] = "false"
            think = f"<think>\n{data['model_reasoning'].strip()}\n</think>"
        else:
            think = "<think>\n</think>"
        answer = f"<answer>\n{data['model_answer'].strip()}\n</answer>"

        history.append({"role": "user", "content": data["question"]})
        history.append({"role": "assistant", "content": think + "\n" + answer})

        data["history"] = history
        data['source'] = "GeneralReasoning/GeneralThought-195K"
        return data


    genreason_dataset = load_dataset("GeneralReasoning/GeneralThought-195K")["train"]
    genreason_dataset = genreason_dataset.filter(
        lambda x: x["question_license"] in ["MIT", "Apache-2.0"]
    )
    # genreason_dataset = genreason_dataset.filter(lambda x: x['task'] in ['Open Conversations', 'Explanation'])
    # genreason_dataset = genreason_dataset.filter(lambda x: get_ascii(x['question']) if x['question'] else False)
    # genreason_dataset = genreason_dataset.filter(
    #     lambda x: get_ascii(x["model_answer"]) if x["model_answer"] else False
    # )
    genreason_dataset = genreason_dataset.filter(
        lambda x: len(x["model_reasoning"].strip().split()) < reasoning_len
        if x["model_reasoning"]
        else True
    )
    # genreason_dataset = genreason_dataset.filter(lambda x: len(x['question'].strip().split()) < 256 if x['question'] else False)

    genreason_dataset = genreason_dataset.map(generalreason_conv)
    delete_keys = list(genreason_dataset.column_names)
    genreason_dataset = genreason_dataset.map(
        lambda x: {
            "conversations": tokenizer.apply_chat_template(
                x["history"], tools=None, tokenize=False
            )
        }
    )
    genreason_dataset = genreason_dataset.remove_columns(delete_keys)
    # genreason_dataset = genreason_dataset.select(range(150))
    print("General reason dataset length (after filter):", len(genreason_dataset))


def process(data):
    for idx, message in enumerate(data["messages"]):
        if message["role"] != "assistant":
            continue
        content = message["content"]
        tag = "</think>"
        pos = content.find(tag)
        answer = content[pos + len(tag) :].strip()
        data["messages"][idx]["content"] = (
            content[:pos].strip() + f"\n</think>\n<answer>\n{answer}\n</answer>"
        )
    data['source'] = "open-r1/codeforces-cots/solutions_py_decontaminated"
    return data

# Login using e.g. `huggingface-cli login` to access this dataset
codeforces_cot = load_dataset(
    "open-r1/codeforces-cots", "solutions_py_decontaminated"
)["train"]
# codeforces_cot = codeforces_cot.filter(lambda x: len(str(x["messages"])) < 8000)
delete_keys = list(codeforces_cot.column_names)
codeforces_cot = codeforces_cot.map(process)
codeforces_cot = codeforces_cot.map(
    lambda x: {
        "conversations": tokenizer.apply_chat_template(
            x["messages"], tools=None, tokenize=False
        )
    }
)
codeforces_cot = codeforces_cot.remove_columns(delete_keys)
# codeforces_cot = codeforces_cot.select(range(150))
print("Codeforces CoT dataset length (after filter):", len(codeforces_cot))


# if not dataset:
# if False:
#     def process(data):
#         seq = [
#             {"role": "user", "content": data["question"]},
#             {
#                 "role": "assistant",
#                 "content": f"<think>\n</think>\n<tool_call>\n{{'name': 'web_search', 'arguments': {{'search_str': '{data['search_str']}'}}}}</tool_call>",
#             },
#             {
#                 "role": "tool",
#                 "content": data["search_results"]
#                 + f"\n\n\nUser question: {data['question']}\n",
#             },
#             {
#                 "role": "assistant",
#                 "content": f"<think>{data['think'].strip()}</think>\n<answer>\n{data['answer'].strip()}\n</answer>",
#             },
#         ]

#         # data["messages"] = seq
#         data["conversations"] = tokenizer.apply_chat_template(
#             seq, tools=[webtool_def], tokenize=False
#         )
#         return data

#     websearch_data = []
#     with open("datagen/search_data.jsonl", "r") as f:
#         for line in f:
#             websearch_data.append(json.loads(line))
#     websearch_data = Dataset.from_list(websearch_data)
#     delete_keys = list(websearch_data.column_names)
#     websearch_data = websearch_data.map(process)
#     websearch_data = websearch_data.remove_columns(delete_keys)
#     print("WebSearch dataset length:", len(websearch_data))


if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

dataset_list = [
    ("smoltalk oss", hf_oss),
    ("R1", r1_dataset),
    ("Function Calling", fc_dataset),
    ("General Reason", genreason_dataset),
    ("Codeforce CoT", codeforces_cot),
    ("smoltalk fc", smoltalk_fc_dataset),
    ("smoltalk constraints", hf_constraints),
    ("smoltalk everyday_conv", hf_everyday_conv),
    ("smoltalk syschats", hf_systemchats),
    # ("WebSearch", websearch_data),
]

with open(os.path.join(SAVE_PATH, "dataset_example.log"), "w") as f:
    for k, d in dataset_list:
        f.write(f"\n{k} length: {len(d)}\n")
        f.write(f"{k}\n{'-' * 20}\n{d[0]['conversations']}\n{'=' * 20}\n\n")

dataset = concatenate_datasets([e[1] for e in dataset_list])
dataset = dataset.shuffle(12)#.select(range(500_000))
# dataset = concatenate_datasets([hf_oss.select(range(20_000)), dataset])
selected_len = []

def token_len_mapper(data):
    d = tokenizer.encode(data["conversations"])
    data['token_len'] = len(d)
    return data

# dataset = dataset.select(selected_len)
dataset = dataset.map(token_len_mapper)
dataset = dataset.filter(lambda x: x['token_len'] < CONTEXT_LEN).select(range(300_000))
dataset.to_json(f"datasets/dataset_ctx{CONTEXT_LEN}_cot{REASONING_LEN}.jsonl", orient="records")
# dataset.save_to_disk(f"datasets/dataset_ctx{CONTEXT_LEN}_cot{REASONING_LEN}")