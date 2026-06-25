import anthropic
import os

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

message = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[
        {"role": "user", 
        "content": "Hello Claude! I just started my FDE journey. Give me one piece of advice."}
    ]
)

print(message.content[0].text)