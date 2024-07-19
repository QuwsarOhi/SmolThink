from jinja2 import Template

# Example context similar to the Go template context
context = {
    "Messages": [
        {"Role": "user", "Content": "User message", "System": True, "Tools": "some_tool"},
        {"Role": "assistant", "Content": "Assistant response", "ToolCalls": [{"Function": {"Name": "tool_func", "Arguments": ["arg1", "arg2"]}}]},
        {"Role": "tool", "Content": "Tool result"}
    ],
    "Prompt": "Initial prompt",
    "Response": "Final response"
}

# Define the template string
template_str = """
{% if Messages %}
  {% for index, message in enumerate(Messages) %}
    {% if message.Role == "user" %}
      {% if (Messages[index:] | length == 1) and Tools %}[AVAILABLE_TOOLS] {{ Tools }} [/AVAILABLE_TOOLS]{% endif %}
      [INST] {% if System and (Messages[index:] | length == 1) %}{{ System }}{% endif %}
{{ message.Content }} [/INST]
    {% elif message.Role == "assistant" %}
      {% if message.Content %} {{ message.Content }}
      {% elif message.ToolCalls %}[TOOL_CALLS] [
        {% for call in message.ToolCalls %}{"name": "{{ call.Function.Name }}", "arguments": {{ call.Function.Arguments | tojson }}}
        {% endfor %}
      ]
      {% endif %}
    {% elif message.Role == "tool" %}
      [TOOL_RESULTS] {"content": {{ message.Content | tojson }}} [/TOOL_RESULTS]
    {% endif %}
  {% endfor %}
{% else %}
  [INST] {% if System %}{{ System }}{% endif %} {{ Prompt }} [/INST]
{% endif %} {{ Response }}
{% if Response %}</s>{% endif %}
"""

# Create a Jinja2 template object
template = Template(template_str)

# Render the template with the provided context
rendered_output = template.render(context)

print(rendered_output)
