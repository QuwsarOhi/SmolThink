import openai

def get_gorilla_response(prompt="", model="gorilla-openfunctions-v2", functions=[]):
    openai.api_key = "EMPTY"  # Hosted for free with ❤️ from UC Berkeley
    openai.api_base = "http://luigi.millennium.berkeley.edu:8000/v1"
    try:
        completion = openai.ChatCompletion.create(
            model="gorilla-openfunctions-v2",
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            functions=functions,
        )
        # completion.choices[0].message.content, string format of the function call 
        # completion.choices[0].message.functions, Json format of the function call
        return completion.choices[0]
    except Exception as exp:
        print(exp)


query = "What's the weather like in the two cities of Boston and San Francisco?"

functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]

print(get_gorilla_response(prompt=query, functions=[functions]))