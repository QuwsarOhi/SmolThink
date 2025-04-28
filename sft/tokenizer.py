import json
from transformers import AutoTokenizer

CHAT_TEMPLATE = """<empty_output>{% for message in messages %}
{% if loop.first and messages[0]['role'] != 'system' %}
{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM<|im_end|>' }}
{% endif %}
{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>'}}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|im_start|>assistant' }}
{% endif %}"""


TOOL_TEMPLATE = """You are a helpful AI assistant. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the functions can be used, point it out and refuse to answer. 
If the given question lacks the parameters required by the function, also point it out.

You have access to the following tools:
<tools>{tools}</tools>

You first think/plan inside <think></think> tags.
Then for each function call, return a json list object with function name and arguments within <tool_call></tool_call> tags."""


def get_tokenizer(model_path):
    """
    Initializes and configures a tokenizer using a pre-trained model.

    Args:
        model_path (str): The path to the pre-trained model to load the tokenizer from.

    Returns:
        AutoTokenizer: A tokenizer instance configured with specific tokens and attributes.

    Notes:
        - The tokenizer is initialized with a custom chat template and specific token values:
            - `bos_token` (beginning-of-sequence token): "<empty_output>"
            - `eos_token` (end-of-sequence token): "<|im_end|>"
            - `pad_token` (padding token): "<|endoftext|>"
            - `unk_token` (unknown token): "<|endoftext|>"
        - The function includes assertions to ensure the token IDs match expected values:
            - `bos_token_id` should be 16.
            - `eos_token_id` should be 2.
            - `pad_token_id` should be 0.
            - `unk_token_id` should be 0.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.bos_token = "<empty_output>"
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.unk_token = "<|endoftext|>"
    # tokenizer.padding_side = "left"
    # tokenizer.truncation_side = "left"

    assert tokenizer.bos_token_id == 16
    assert tokenizer.eos_token_id == 2
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 0

    return tokenizer


if __name__ == '__main__':
    SIZE = "360M"
    MODEL_PATH = f"HuggingFaceTB/SmolLM2-{SIZE}"

    tokenizer = get_tokenizer(MODEL_PATH)

    print(
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I am fine"},
            ],
            tokenize=False,
            add_generation_prompt=True
        )
    )

    ## ----- Prompt Template Debugging ------
    tools = [
        {
            "type": "function",
            "function": {
                "name": "retrieve_payment_status",
                "description": "Get payment status of a transaction",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "transaction_id": {
                            "type": "string",
                            "description": "The transaction id.",
                        }
                    },
                    "required": ["transaction_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "retrieve_payment_date",
                "description": "Get payment date of a transaction",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "transaction_id": {
                            "type": "string",
                            "description": "The transaction id.",
                        }
                    },
                    "required": ["transaction_id"],
                },
            },
        },
    ]
    print("\n-----\n")
    print(tokenizer.apply_chat_template([
        {"role": "system", "content": TOOL_TEMPLATE.format(tools=json.dumps(tools, indent=2))},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "<tool_call>[retrieve_payment_date(12)]</tool_call>"},
        {"role": "tool", "content": "12/12/12"},
        {"role": "assistant", "content": "12/12/12"}
    ], tokenize=False))