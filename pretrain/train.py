import torch
import sys
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer,
    TrainingArguments,
)


SIZE = "360M"
CONTEXT_LEN = 1024
CONTEXT_STRIDE = 2
SAVE_STEPS = 20
MODEL_PATH = f"HuggingFaceTB/SmolLM2-{SIZE}"

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     device_map="cpu",
#     low_cpu_mem_usage=True,
#     attn_implementation="eager",
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     use_cache=True,
#     tie_word_embeddings=True,
# ).to("mps")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# Smollm2 did not have bos token
# tokenizer.bos_token = "<empty_output>"
# Smollm2 did not have eos token
# tokenizer.eos_token = "<|im_end|>"
tokenizer.pad_token = "<|endoftext|>"
tokenizer.unk_token = "<|endoftext|>"
tokenizer.add_bos_token = False
tokenizer.add_eos_token = False


# Check if no special tokens are being added
s = "This is a string"
assert s == tokenizer.decode(tokenizer.encode(s), skip_special_tokens=False), "Problem with tokenizer"
sys.exit()


### Load dataset here
train_ds = None


class DatasetGen_v1(torch.utils.data.Dataset):
    ''' 
    Memory optimized dataset. Only converts into tokens when necessary. 
    Context stride helps the LLM to go through the sequence
    '''
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.cache = None
        self.cache_idx = -1
        self.cache_len = 0
        self.indices = [(i, 0) for i in range(len(self.dataset))]
        # self._get_len()

    def _get_len(self):
        print("Computing dataset length")
        for idx in tqdm(range(len(self.dataset))):
            self.gen(idx)
            for i in range(self.cache_len):
                self.indices.append((idx, i))
        print("Total length of data:", len(self.indices))

    def __len__(self):
        return len(self.indices)

    def gen(self, idx):
        self.cache = self.dataset[idx]["conversations"].rstrip()
        self.cache = self.tokenizer(
            self.cache,
            max_length=CONTEXT_LEN,
            truncation=True,
            return_overflowing_tokens=True,  # Return the overflowing tokens
            stride=CONTEXT_LEN // CONTEXT_STRIDE,
            padding="max_length",
        )
        self.cache_idx = idx
        self.cache_len = len(self.cache["input_ids"])

    def __getitem__(self, idx):
        p, q = self.indices[idx]
        if self.cache_idx != p:
            self.gen(p)

        input_ids = self.cache["input_ids"][q]
        attention_mask = self.cache["attention_mask"][q]
        labels = [-100 if t == self.tokenizer.pad_token_id else t for t in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# REF: https://github.com/huggingface/smollm/blob/main/text/pretraining/smollm2/config_smollm2_360M.yaml

training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    
    learning_rate= 0.003 / 6,
    lr_scheduler_type="linear",
    adam_beta1=0.9,
    adam_beta2=0.95,
    optim="adamw_torch",    # adamw_torch, adafactor
    weight_decay=0.01,
    max_grad_norm=1.0,      # Reduce to 0.1 if NaN
    warmup_ratio= 0.1,

    num_train_epochs=1,
    logging_strategy="steps",
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    save_strategy="steps",  #'steps', 'no', 'best',
    logging_steps=20,
    max_steps=len(train_ds),
    save_steps=SAVE_STEPS,
    save_total_limit=3,

    # Memory reduction
    # gradient_checkpointing=True,
    # gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    bf16_full_eval=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,      # Increase to 4 for smoother training
    torch_empty_cache_steps=SAVE_STEPS,
    push_to_hub=False,
    report_to="none",
    dataloader_pin_memory=True,

    # Speedups
    torch_compile=True,
    torch_compile_backend='aot_eager'
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    data_collator=None,
)

trainer.train()