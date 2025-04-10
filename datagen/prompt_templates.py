# Phi3 templates

search_query_template_phi3 = \
"""<|system|>
You are a helpful assistant. You will be asked a question. Imagine you do not know the answer of the question. As a result, you want to search the web to find the answer. For the given user question, write a search question that could be used in search engine to find the answer to the question. The search string that you would produce should be inside <search tag>. 
For example: <search> your search string </search><|end|>
<|user|>
{query}<|end|>
<|assistant|>
Sure! Here is the one short search string that I would use to search on the web:
<search>"""


question_gen_template_phi3 = \
"""<|system|>
You are a helpful assistant who is very good at generating question. The user would give you a topic, you have to come up with a question so that the question can be used to train a good quality LLM.<|end|>
<|user|>
Generate question in topic of: {topic_name}

Produce the question inside "question" tag. Example: <question> your generated question </question><|end|>
<|assistant|>
Sure! Here is the question on topic {topic_name}:
<question>"""

# # Ref: https://huggingface.co/learn/cookbook/en/llm_judge
llm_judge_prompt_phi3 = \
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
- Give 1 additional point if the answer is clear, precise, and not repetitive.
- Provide 1 further point if the answer is true and follows proper markdown format.
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

grpo_judge_prompt_phi3 = \
"""<|system|>
You will be given a user_question and search_string couple.
Your task is to provide a 'total rating' scoring how well the search_string reflects user_question so that the search_sting can be used in a web search engine.
Give your answer on a scale of 1 to 3, where 1 means that the search_string is not helpful at all, and 3 means that the search_string completely reflects user_question and higher chance of getting correct answer through web search.

Here is the scale you should use to build your answer:
1: The search_string is terrible: completely irrelevant to the question asked
2: The search_string is mostly not helpful: misses some key aspects of the question, incomplete
3: The search_string is excellent: relevant, direct, and addresses all the concerns raised in the question

Use the following rubrics to award the points:
- Award 1 point if the search_string is completely irrelevant 
- Award 1 point if the search_string is somewhat related to the user_question.
- One final point should be awarded if the search_string consists all key information from user_question to do a web search.


Provide your feedback as follows:

Feedback:::
Evaluation: <eval> (your rationale for the rating, as a text) </eval>
Total rating: <rating> (your rating, as a number between 1 and 3) </rating>

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.<|end|>
<|user|>
Now here are the user_question and search_string.

user_question: {user_question}
search_string: {search_string}

Provide your feedback. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.<|end|>
<|assistant|>
Feedback:::
Evaluation: <eval>"""


# %%

tool_call_template = \
"""You are a helpful assistant. When using a tool, format as:
<tool_call>
{{'name': 'function_name', 'parameters': {{'param1': 'value1'}}}}
</tool_call>

The following tool for all tasks:
{{"type": "function", "function": {{"name": "web_search", "description": "Can search the web for infomation which are doubtful/unknown/recent.", "parameters": {{"type": "object", "properties": {{"search_str": {{"type": "string", "description": "The whole question you want to ask.", "required": true}}}}}}}}}}

Given the tools, please respond with a JSON object for a function call with its proper arguments that best answers the given prompt.
Prioritize specific concurrent tool calls over a single generic tool call. Use tools first, explain what happened after recieving the results.
Respond in the format {{"name": function name, "parameters": dictionary of argument name and its value}}.

<｜User｜>
Who is the current president of USA?
<｜end▁of▁sentence｜>
<｜Assistant｜>
<think>
So the user is asking "Who is the current president of USA?". Looking at the available tool, I see I have access to "web_search" which can be used to search information in web which are recent or unknown or doubtful. As the question requires recent knowledge, I would use "web_search". 

It requires a default parameter "search_str". Based on the user question, the "search_str" should be "current president of USA"
</think>

<tool_call>
{{'name': 'web_search', 'arguments': {{'query': 'current president of USA'}}}}
</tool_call>
<｜end▁of▁sentence｜>
<｜User｜>
{query}
<｜end▁of▁sentence｜>
<｜Assistant｜>
<think>
Let's check the available tools as the user has asked me to use it. """

search_query_template_deepseek = \
"""You are a helpful assistant. You will be asked a question. Imagine you do not know the answer of the question. As a result, you want to search the web to find the answer. For the given user question, write a search question that could be used in search engine to find the answer to the question. The search string that you would produce should be inside <search tag>. 
For example: <search> your search string </search>
<｜User｜>
{query}
<｜end▁of▁sentence｜>
<｜Assistant｜>
<think>

</think>

Sure! Here is the one short search string that I would use to search on the web:
<search>"""


casual_conv_template = \
"""You are Jhon Doe, a mid-aged person who lives in a peaceful country. You love to casually chat with friends and family and enjoy asking casual questions about life, weather, sunshine, breeze, and traffic.

<｜User｜>
You are going to start conversation with a child. Write a conversation starter question does not require any information to convey. 

Produce the question inside "question" tag. Example: <question> your generated question </question>
<｜end▁of▁sentence｜>
<｜Assistant｜>
<think>

</think>

Sure! Here is the question:
<question>"""


question_gen_template_deepseek = \
"""You are a helpful assistant who is very good at generating question. The user would give you a topic, you have to come up with a question so that the question can be used to train a good quality LLM.
<｜User｜>
Generate question in topic of: {topic_name}

Produce the question inside "question" tag. Example: <question> your generated question </question>
<｜end▁of▁sentence｜>
<｜Assistant｜>
<think>

</think>

Sure! Here is the question on topic {topic_name}:
<question>"""


# Ref: https://huggingface.co/learn/cookbook/en/llm_judge
llm_judge_prompt_deepseek = \
"""You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpful at all, and 4 means that the system_answer completely and helpfully addresses the user_question.

Here is the scale you should use to build your answer:
1: The system_answer is terrible: completely irrelevant to the question asked, or very partial
2: The system_answer is mostly not helpful: misses some key aspects of the question
3: The system_answer is mostly helpful: provides support, but still could be improved
4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question

Use the following rubrics to award the points:
- Award 1 point if the answer is related to the question.
- Give 1 additional point if the answer is clear, precise, and not repetitive.
- Provide 1 further point if the answer is true and follows proper markdown format.
- One final point should be awarded if the answer provides additional resources to support the user.


Provide your feedback as follows:

Feedback:::
Evaluation: <eval> (your rationale for the rating, as a text) </eval>
Total rating: <rating> (your rating, as a number between 1 and 4) </rating>

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.<｜end▁of▁sentence｜>
<｜User｜>
Now here are the question and answer.

Question: {question}
Answer: {answer}

Provide your feedback. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.<｜end▁of▁sentence｜>
<｜Assistant｜>
<think>

</think>

Feedback:::
Evaluation: <eval>"""


deepseek_r1_context = '''You are a helpful AI. You will be given a question and a context. You have to first think (in 150 words) and then come up with the final answer based on user question and user-provided context. Do not provide information outside the given context. Strictly answer the question based on the given context. Do not giveaway/mix information from your knowledge.

<｜User｜>
Use the following content to answer user question:
{context}

User question: {question}
<｜end▁of▁sentence｜>
<｜Assistant｜>
<think>
I will finish thinking in at most 150 words. Now let's think. '''

deepseek_r1_nocontext = '''You are a helpful AI named SmolThink. You have a delightful personality who loves to assist people. Given a question, come up with a calm and helpful response. 

<｜User｜>{question}
<｜end▁of▁sentence｜>
<｜Assistant｜>
<think>

</think>

'''