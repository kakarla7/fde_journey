import streamlit as st
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

SYSTEM_PROMPT = """You are an expert travel planning agent. When given a destination 
and trip length, you use your tools to gather weather, attractions, and travel tips, 
then build a detailed day-by-day itinerary. Always be specific, practical and exciting.
Structure your final response as:

## ✈️ Trip Overview
[brief summary]

## 📅 Day-by-Day Itinerary
[detailed plan for each day]

## 💡 Next Steps
[exactly 5 actionable next steps the traveller should do NOW to prepare]"""

tools = [
    {
        "name": "get_weather",
        "description": "Returns typical weather for a city in a given month",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The destination city"},
                "month": {"type": "string", "description": "Month of travel e.g. October"}
            },
            "required": ["city", "month"]
        }
    },
    {
        "name": "get_attractions",
        "description": "Returns top attractions and activities for a city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The destination city"},
                "trip_length": {"type": "integer", "description": "Number of days"}
            },
            "required": ["city", "trip_length"]
        }
    },
    {
        "name": "get_travel_tips",
        "description": "Returns practical travel tips for a city including transport, currency, and local customs",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The destination city"}
            },
            "required": ["city"]
        }
    }
]

def run_tool(tool_name, tool_input):
    if tool_name == "get_weather":
        city = tool_input.get("city", "")
        month = tool_input.get("month", "")
        return f"Weather in {city} in {month}: typically mild to warm (18-26°C), low chance of rain. Pack light layers. Best time to visit outdoor attractions."

    elif tool_name == "get_attractions":
        city = tool_input.get("city", "")
        days = tool_input.get("trip_length", 3)
        return f"Top attractions in {city} for {days} days: Historic old town, Local food markets, Museums and galleries, Day trip to nearby nature spots, Rooftop bars and local restaurants, Cultural landmarks, Hidden neighbourhood gems."

    elif tool_name == "get_travel_tips":
        city = tool_input.get("city", "")
        return f"Travel tips for {city}: Use public transport (metro/bus), local currency preferred at markets, tip 10-15% at restaurants, book popular attractions in advance, learn 5 basic phrases in local language, stay in central neighbourhoods for best access."

    return "Tool not found"

def serialize_content(content_blocks):
    result = []
    for block in content_blocks:
        if block.type == "text":
            result.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            result.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input
            })
    return result

def get_travel_plan(destination, trip_length, month):
    if "api_history" not in st.session_state:
        st.session_state.api_history = []

    user_message = f"Plan me a {trip_length} day trip to {destination} in {month}."

    st.session_state.api_history.append({
        "role": "user",
        "content": user_message
    })

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=tools,
        messages=st.session_state.api_history
    )

    tools_used = []

    while response.stop_reason == "tool_use":
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        st.session_state.api_history.append({
            "role": "assistant",
            "content": serialize_content(response.content)
        })

        tool_results = []
        for block in tool_use_blocks:
            tools_used.append(block.name)
            result = run_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result
            })

        st.session_state.api_history.append({
            "role": "user",
            "content": tool_results
        })

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=st.session_state.api_history
        )

    final_text = response.content[0].text
    st.session_state.api_history.append({
        "role": "assistant",
        "content": final_text
    })

    return final_text, tools_used


# ── Streamlit UI ───────────────────────────────────────────────
st.set_page_config(page_title="AI Travel Agent", page_icon="✈️", layout="wide")

if "api_history" not in st.session_state:
    st.session_state.api_history = []

if "plan" not in st.session_state:
    st.session_state.plan = None

if "tools_used" not in st.session_state:
    st.session_state.tools_used = []

with st.sidebar:
    st.title("✈️ AI Travel Agent")
    st.caption("Powered by Claude")
    st.divider()

    destination = st.text_input("🌍 Destination", placeholder="e.g. Tokyo, Paris, Bali")
    month = st.selectbox("📅 Month of travel", [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ])
    trip_length = st.slider("🗓️ Trip length (days)", 1, 14, 3)

    if st.button("🚀 Plan my trip", use_container_width=True):
        if destination:
            st.session_state.api_history = []
            st.session_state.plan = None
            st.session_state.tools_used = []
            with st.spinner("Planning your trip..."):
                plan, tools_used = get_travel_plan(destination, trip_length, month)
                st.session_state.plan = plan
                st.session_state.tools_used = tools_used
        else:
            st.warning("Please enter a destination!")

    st.divider()

    if st.session_state.tools_used:
        st.markdown("**🔧 Tools used**")
        for tool in st.session_state.tools_used:
            st.success(tool)

    if st.button("🗑️ Clear plan", use_container_width=True):
        st.session_state.api_history = []
        st.session_state.plan = None
        st.session_state.tools_used = []
        st.rerun()

# ── Main area ──────────────────────────────────────────────────
st.title("✈️ AI Travel Planner")
st.caption("Enter your destination and trip details in the sidebar to get started")

if st.session_state.plan:
    # Split plan and next steps into two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(st.session_state.plan)

    with col2:
        st.subheader("📋 Your Next Steps")
        st.info("""
        Once your plan is ready, your next steps 
        appear in the itinerary above under 
        **💡 Next Steps** — scroll down to see them!
        """)
        st.divider()
        st.markdown("**🔁 Want to refine?**")
        st.markdown("- Change the month and replan")
        st.markdown("- Adjust trip length")
        st.markdown("- Try a different destination")

else:
    st.markdown("""
    ### 👋 Welcome to your AI Travel Planner!
    
    **How it works:**
    1. Enter your destination in the sidebar
    2. Pick your travel month and trip length
    3. Hit **Plan my trip** 🚀
    4. Get a full day-by-day itinerary + next steps
    
    **Powered by 3 AI tools:**
    - 🌤️ Weather checker
    - 🗺️ Attractions finder  
    - 💡 Local travel tips
    """)