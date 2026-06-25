import anthropic
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── System prompt ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are FDE Coach, a sharp and encouraging AI assistant 
helping Mounika become a Forward Deployed Engineer specializing in AI and 
agentic systems. You are concise, practical, and always tie advice back to 
real FDE skills. When asked the time, use the get_current_time tool."""

# ── Tool definition ────────────────────────────────────────────
tools = [
    {
        "name": "get_current_time",
        "description": "Returns the current date and time",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
     {
        "name": "get_calendar_date",
        "description": "Returns today's date, day of the week, and how many days are left in the current month",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

# ── Tool execution ─────────────────────────────────────────────
def run_tool(tool_name, tool_input):
    if tool_name == "get_current_time":
        return datetime.now().strftime("%A, %B %d %Y — %I:%M %p")
    
    elif tool_name == "get_calendar_date":
        now = datetime.now()
        if now.month == 12:
            last_day = 31
        else:
            last_day = (datetime(now.year, now.month + 1, 1) - timedelta(days=1)).day
        days_left = last_day - now.day
        
        return (
            f"Today is {now.strftime('%A, %B %d %Y')}. "
            f"There are {days_left} days left in {now.strftime('%B')}."
        )
    
    else:
        return "Tool not found"
# ── Chat function ──────────────────────────────────────────────
def chat(conversation_history, user_message):
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=tools,
        messages=conversation_history
    )

    # Handle tool use
    if response.stop_reason == "tool_use":
        tool_use_block = next(b for b in response.content if b.type == "tool_use")
        print(f"🔧 Tool called: {tool_use_block.name}")
        tool_result = run_tool(tool_use_block.name, tool_use_block.input)
        print(f"🔧 Tool result: {tool_result}")

        # Add assistant + tool result to history
        conversation_history.append({"role": "assistant", "content": response.content})
        conversation_history.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": tool_result
            }]
        })

        # Get final response after tool use
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=conversation_history
        )

    assistant_message = response.content[0].text
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    return assistant_message, conversation_history

# ── Main loop ──────────────────────────────────────────────────
def main():
    print("\n🤖 FDE Coach is ready! Type 'quit' to exit.\n")
    conversation_history = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\nFDE Coach: Keep building — see you tomorrow! 🚀\n")
            break

        response, conversation_history = chat(conversation_history, user_input)
        print(f"\nFDE Coach: {response}\n")

if __name__ == "__main__":
    main()