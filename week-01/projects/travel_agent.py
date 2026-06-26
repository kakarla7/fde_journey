import streamlit as st
import anthropic
import os
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

SYSTEM_PROMPT = """You are an expert travel planning agent. When given a destination and trip length, 
you use your tools to gather weather, attractions, and travel tips, then build a detailed day-by-day plan.

You MUST respond with ONLY a valid JSON object — no markdown, no backticks, no explanation. 
Respond with exactly this structure:

{
  "destination": "city name",
  "duration": 3,
  "month": "October",
  "weather_summary": "brief weather description",
  "days": [
    {
      "day": 1,
      "theme": "short day theme e.g. Arrival & Old Town",
      "attractions": [
        {"name": "Place name", "time": "Morning", "area": "Neighbourhood", "emoji": "⛩️", "description": "one line description"}
      ],
      "food": [
        {"name": "Restaurant name", "meal": "Breakfast/Lunch/Dinner", "description": "one line, cuisine type"}
      ],
      "stay": [
        {"name": "Hotel name", "tier": "Budget/Mid-range/Luxury", "area": "Neighbourhood"}
      ]
    }
  ],
  "next_steps": [
    "Book flights at least 6 weeks in advance",
    "Reserve hotel in central neighbourhood",
    "Get local SIM card or international plan",
    "Download offline maps",
    "Book popular attractions in advance"
  ],
  "what_to_wear": [
    "Light layers for mild weather",
    "Comfortable walking shoes",
    "Rain jacket just in case",
    "Smart casual for restaurants"
  ],
  "currency": {
    "local_currency": "Japanese Yen (JPY)",
    "exchange_tip": "Exchange at airport or use ATMs — avoid hotel exchange",
    "daily_budget_estimate": "$80-150 USD per day",
    "cash_or_card": "Cash preferred at smaller restaurants and temples",
    "useful_amounts": "Keep 5000-10000 JPY in cash daily"
  },
  "travel_tips": [
    "Get an IC card for all public transport",
    "Learn 5 basic phrases in local language",
    "Tip: 10-15% at sit-down restaurants"
  ]
}"""

tools = [
    {
        "name": "get_weather",
        "description": "Returns typical weather for a city in a given month",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "month": {"type": "string"}
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
                "city": {"type": "string"},
                "trip_length": {"type": "integer"}
            },
            "required": ["city", "trip_length"]
        }
    },
    {
        "name": "get_travel_tips",
        "description": "Returns practical travel tips including transport, currency, customs, and what to wear",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
]

def run_tool(tool_name, tool_input):
    if tool_name == "get_weather":
        city = tool_input.get("city", "")
        month = tool_input.get("month", "")
        return f"Weather in {city} in {month}: mild to warm 18-26C, low chance of rain, occasional cool evenings. Light layers recommended."
    elif tool_name == "get_attractions":
        city = tool_input.get("city", "")
        days = tool_input.get("trip_length", 3)
        return f"Top attractions in {city} for {days} days: historic temples and shrines, local food markets, art museums, scenic parks, rooftop bars, cultural landmarks, hidden neighbourhood gems, day trip options nearby."
    elif tool_name == "get_travel_tips":
        city = tool_input.get("city", "")
        return f"Tips for {city}: use public metro/bus, carry local cash for markets, book popular spots in advance, learn basic local phrases, respect local customs, central neighbourhoods best for access, tipping culture varies."
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

    st.session_state.api_history.append({
        "role": "user",
        "content": f"Plan me a {trip_length} day trip to {destination} in {month}."
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

    print("=== RAW CLAUDE RESPONSE ===")
    print(repr(final_text))
    print("=== END ===")

    start = final_text.find("{")
    end = final_text.rfind("}") + 1

    if start == -1:
        raise ValueError("No JSON found in Claude response")

    clean = final_text[start:end]
    plan = json.loads(clean)
    return plan, tools_used


# ── Streamlit UI ───────────────────────────────────────────────
st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")

if "api_history" not in st.session_state:
    st.session_state.api_history = []
if "plan" not in st.session_state:
    st.session_state.plan = None
if "tools_used" not in st.session_state:
    st.session_state.tools_used = []
if "active_day" not in st.session_state:
    st.session_state.active_day = 0
if "checked_steps" not in st.session_state:
    st.session_state.checked_steps = {}

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.title("✈️ AI Travel Planner")
    st.caption("Powered by Claude")
    st.divider()

    destination = st.text_input("🌍 Destination", placeholder="e.g. Tokyo, Paris, Bali")
    month = st.selectbox("📅 Month of travel", [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ])
    trip_length = st.slider("🗓️ Trip length (days)", 1, 7, 3)

    if st.button("🚀 Plan my trip", use_container_width=True):
        if destination:
            st.session_state.api_history = []
            st.session_state.plan = None
            st.session_state.tools_used = []
            st.session_state.active_day = 0
            st.session_state.checked_steps = {}
            with st.spinner("Planning your trip..."):
                try:
                    plan, tools_used = get_travel_plan(destination, trip_length, month)
                    st.session_state.plan = plan
                    st.session_state.tools_used = tools_used
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
        else:
            st.warning("Please enter a destination!")

    if st.session_state.tools_used:
        st.divider()
        st.markdown("**🔧 Tools used**")
        for tool in st.session_state.tools_used:
            st.success(tool)

    st.divider()
    st.markdown("**🚧 Coming soon**")
    coming_soon = ["🌤️ Real weather API", "🗺️ Google Places", "✈️ Flight search", "🏨 Hotel booking", "💱 Live exchange rates"]
    for item in coming_soon:
        st.caption(item)

    if st.session_state.plan:
        st.divider()
        if st.button("🗑️ Clear plan", use_container_width=True):
            st.session_state.api_history = []
            st.session_state.plan = None
            st.session_state.tools_used = []
            st.session_state.active_day = 0
            st.session_state.checked_steps = {}
            st.rerun()

# ── Main area ──────────────────────────────────────────────────
st.title("✈️ AI Travel Planner")
st.caption("Enter your destination in the sidebar to get started")

if not st.session_state.plan:
    st.markdown("""
    ### 👋 Welcome!
    **How it works:**
    1. Enter your destination in the sidebar
    2. Pick your travel month and trip length
    3. Hit **Plan my trip** 🚀
    4. Get a full day-by-day itinerary with attractions, food, stay, what to wear, and currency tips!
    
    **Powered by 3 AI tools:**
    - 🌤️ Weather checker
    - 🗺️ Attractions finder
    - 💡 Local travel tips
    """)
else:
    plan = st.session_state.plan

    # ── Trip header ──
    st.subheader(f"📍 {plan.get('destination')} · {plan.get('duration')} days · {plan.get('month')}")
    st.info(f"🌤️ {plan.get('weather_summary', '')}")

    # ── Day tabs ──
    days = plan.get("days", [])
    tab_labels = [f"Day {d['day']} — {d['theme']}" for d in days]
    tabs = st.tabs(tab_labels)

    for i, tab in enumerate(tabs):
        with tab:
            day = days[i]
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("##### 🗺️ Attractions")
                for a in day.get("attractions", []):
                    with st.container():
                        st.markdown(f"""
<div style="background:var(--background-color);border:1px solid #e0e0e0;border-radius:10px;padding:10px;margin-bottom:8px">
<b>{a.get('emoji','')} {a.get('name')}</b><br>
<small>🕐 {a.get('time')} · 📍 {a.get('area')}</small><br>
<small style="color:gray">{a.get('description')}</small>
</div>""", unsafe_allow_html=True)

            with col2:
                st.markdown("##### 🍽️ Where to eat")
                for f in day.get("food", []):
                    with st.container():
                        st.markdown(f"""
<div style="background:var(--background-color);border:1px solid #e0e0e0;border-radius:10px;padding:10px;margin-bottom:8px">
<b>{f.get('name')}</b><br>
<small>🍴 {f.get('meal')}</small><br>
<small style="color:gray">{f.get('description')}</small>
</div>""", unsafe_allow_html=True)

                st.markdown("##### 🏨 Where to stay")
                for s in day.get("stay", []):
                    st.markdown(f"""
<div style="background:var(--background-color);border:1px solid #e0e0e0;border-radius:10px;padding:10px;margin-bottom:8px">
<b>{s.get('name')}</b><br>
<small>⭐ {s.get('tier')} · 📍 {s.get('area')}</small>
</div>""", unsafe_allow_html=True)

            with col3:
                st.markdown("##### 💡 Day tips")
                tips = plan.get("travel_tips", [])
                tip_index = i % len(tips) if tips else 0
                if tips:
                    st.info(tips[tip_index])

                st.markdown("##### 👗 What to wear")
                for item in plan.get("what_to_wear", []):
                    st.markdown(f"- {item}")

    # ── Currency section ──
    st.divider()
    st.markdown("### 💱 Currency & Budget")
    currency = plan.get("currency", {})

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Local Currency", currency.get("local_currency", "—"))
    with c2:
        st.metric("Daily Budget", currency.get("daily_budget_estimate", "—"))
    with c3:
        st.metric("Cash or Card?", currency.get("cash_or_card", "—").split(" ")[0])

    st.markdown(f"💡 **Exchange tip:** {currency.get('exchange_tip', '')}")
    st.markdown(f"💰 **Useful amounts:** {currency.get('useful_amounts', '')}")

    # ── Next steps checklist ──
    st.divider()
    st.markdown("### ✅ Your next steps")
    st.caption("Check these off as you go!")

    next_steps = plan.get("next_steps", [])
    cols = st.columns(2)
    for idx, step in enumerate(next_steps):
        key = f"step_{idx}"
        with cols[idx % 2]:
            checked = st.checkbox(step, key=key)

    # Progress bar
    checked_count = sum(1 for idx in range(len(next_steps)) if st.session_state.get(f"step_{idx}", False))
    if next_steps:
        progress = checked_count / len(next_steps)
        st.progress(progress, text=f"{checked_count} of {len(next_steps)} steps done")