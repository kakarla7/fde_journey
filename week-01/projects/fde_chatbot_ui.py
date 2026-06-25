import streamlit as st
import anthropic
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

SYSTEM_PROMPT = """You are FDE Coach, a sharp and encouraging AI assistant 
helping Mounika become a Forward Deployed Engineer specializing in AI and 
agentic systems. You are concise, practical, and always tie advice back to 
real FDE skills."""

tools = [
    {
        "name": "get_current_time",
        "description": "Returns the current time",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "get_current_date",
        "description": "Returns today's date and days left in the month",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    }
]

def run_tool(tool_name, tool_input):
    if tool_name == "get_current_time":
        return datetime.now().strftime("%A, %B %d %Y — %I:%M %p")
    elif tool_name == "get_current_date":
        now = datetime.now()
        if now.month == 12:
            last_day = 31
        else:
            last_day = (datetime(now.year, now.month + 1, 1) - timedelta(days=1)).day
        days_left = last_day - now.day
        return f"Today is {now.strftime('%A, %B %d %Y')}. {days_left} days left in {now.strftime('%B')}."
    return "Tool not found"

def serialize_content(content_blocks):
    """Convert SDK objects to plain dicts, stripping any extra fields like 'caller'."""
    result = []
    for block in content_blocks:
        if block.type == "text":
            result.append({
                "type": "text",
                "text": block.text
            })
        elif block.type == "tool_use":
            result.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input
                # deliberately exclude 'caller' and any other SDK-only fields
            })
    return result

def serialize_content(content_blocks):
    result = []
    for block in content_blocks:
        if block.type == "text":
            result.append({
                "type": "text",
                "text": block.text
            })
        elif block.type == "tool_use":
            result.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input
            })
    return result

def get_response(user_message):
    if "api_history" not in st.session_state:
        st.session_state.api_history = []

    st.session_state.api_history.append({
        "role": "user",
        "content": user_message
    })

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=tools,
        messages=st.session_state.api_history
    )

    while response.stop_reason == "tool_use":
        # Get ALL tool use blocks in this response (not just the first)
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        # Save assistant message with all tool calls
        st.session_state.api_history.append({
            "role": "assistant",
            "content": serialize_content(response.content)
        })

        # Run ALL tools and return ALL results in one message
        tool_results = []
        for tool_use_block in tool_use_blocks:
            tool_result = run_tool(tool_use_block.name, tool_use_block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": tool_result
            })

        st.session_state.api_history.append({
            "role": "user",
            "content": tool_results  # all results in one message
        })

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=st.session_state.api_history
        )

    final_text = response.content[0].text
    st.session_state.api_history.append({
        "role": "assistant",
        "content": final_text
    })
    return final_text


# ── Streamlit UI ───────────────────────────────────────────────
st.set_page_config(page_title="FDE Coach", page_icon="🤖")

# ── Password gate ──────────────────────────────────────────────
password = st.sidebar.text_input("Access code", type="password")
if password != "fdejourney2024":
    st.sidebar.warning("Enter access code to continue")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hey! I'm FDE Coach. I'm here to help you become a Forward Deployed Engineer. What are we working on today?"
    }]

if "api_history" not in st.session_state:
    st.session_state.api_history = []

with st.sidebar:
    st.title("FDE Coach")
    st.caption("Powered by Claude")
    st.divider()
    st.markdown("**Tools active**")
    st.success("get_current_time")
    st.success("get_current_date")
    st.divider()
    st.markdown("**Week 1 — Claude API**")
    st.progress(1/7, text="Day 1 of 7")
    if st.button("Clear chat"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hey! I'm FDE Coach. I'm here to help you become a Forward Deployed Engineer. What are we working on today?"
        }]
        st.session_state.api_history = []
        st.rerun()

st.title("🤖 FDE Coach")
st.caption("Your AI learning assistant for the FDE journey")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask FDE Coach anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})