import streamlit as st
import anthropic
import os
import json
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

SYSTEM_PROMPT = """You are an expert travel planning agent. When given a destination, trip length, month, and any dietary restrictions,
you use your tools to gather weather, attractions, and travel tips, then build a detailed day-by-day plan.

You MUST respond with ONLY a valid JSON object — no markdown, no backticks, no preamble, no explanation.
Start your response with { and end with }. Nothing before or after.

CRITICAL: Every attraction, restaurant and hotel MUST include realistic, accurate lat/lng GPS coordinates.
LIMIT: Max 3 attractions, 2 food, 2 stay options per day to keep response concise.

Respond with exactly this structure:
{
  "destination": "city name",
  "country": "country name",
  "language_code": "ja-JP",
  "duration": 3,
  "month": "October",
  "weather_summary": "brief weather description",
  "airport": {"name": "Main Airport Name", "code": "TYO", "lat": 35.7647, "lng": 140.3864},
  "days": [
    {
      "day": 1,
      "theme": "short day theme",
      "attractions": [
        {"name": "Place name", "time": "Morning", "area": "Neighbourhood", "emoji": "⛩️", "description": "one line", "lat": 35.7148, "lng": 139.7967}
      ],
      "food": [
        {"name": "Restaurant name", "meal": "Breakfast/Lunch/Dinner", "description": "one line, cuisine type", "lat": 35.7148, "lng": 139.7967}
      ],
      "stay": [
        {"name": "Hotel name", "tier": "Budget/Mid-range/Luxury", "area": "Neighbourhood", "lat": 35.7148, "lng": 139.7967}
      ]
    }
  ],
  "next_steps": ["Book flights at least 6 weeks in advance"],
  "what_to_wear": ["Light layers for mild weather"],
  "language_cards": [
    {"english": "Hello", "local": "Konnichiwa", "phonetic": "kon-ni-chi-wa"},
    {"english": "Thank you", "local": "Arigatou", "phonetic": "ah-ree-gah-toh"},
    {"english": "Where is the toilet?", "local": "Toire wa doko?", "phonetic": "toy-reh wa doh-koh"},
    {"english": "How much?", "local": "Ikura?", "phonetic": "ee-koo-rah"},
    {"english": "Help!", "local": "Tasukete!", "phonetic": "tah-soo-keh-teh"},
    {"english": "Good morning", "local": "Ohayou", "phonetic": "oh-hah-yoh"},
    {"english": "Good night", "local": "Oyasumi", "phonetic": "oh-yah-soo-mee"},
    {"english": "Excuse me", "local": "Sumimasen", "phonetic": "soo-mee-mah-sen"},
    {"english": "I don't understand", "local": "Wakarimasen", "phonetic": "wah-kah-ree-mah-sen"},
    {"english": "Delicious!", "local": "Oishii!", "phonetic": "oh-ee-shee"}
  ],
  "packing_list": {
    "clothing": [
      {"item": "T-shirts / tops", "quantity": 4},
      {"item": "Pants / bottoms", "quantity": 3},
      {"item": "Socks", "quantity": 5},
      {"item": "Underwear", "quantity": 5},
      {"item": "Light jacket / cardigan", "quantity": 1},
      {"item": "Comfortable walking shoes", "quantity": 1},
      {"item": "Sandals / casual shoes", "quantity": 1},
      {"item": "Smart casual outfit for dinner", "quantity": 1}
    ],
    "essentials": [
      "Passport + visa (if required)",
      "Travel insurance documents",
      "Power adapter",
      "Portable charger / power bank",
      "Sunscreen SPF 30+",
      "Basic first aid kit",
      "Reusable water bottle",
      "Earphones / headphones",
      "Offline maps downloaded",
      "Local currency (cash)"
    ]
  },
  "currency": {
    "local_currency": "Japanese Yen (JPY)",
    "exchange_tip": "Exchange at airport or use ATMs — avoid hotel exchange",
    "daily_budget_estimate": "$80-150 USD per day",
    "cash_or_card": "Cash preferred at smaller restaurants and temples",
    "useful_amounts": "Keep 5000-10000 JPY in cash daily"
  },
  "travel_tips": ["Get an IC card for all public transport"]
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
        "description": "Returns top attractions, restaurants and hotels for a city with GPS coordinates",
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
        return f"Top attractions in {city} for {days} days with accurate GPS coordinates: historic temples, local food markets, art museums, scenic parks, rooftop bars, cultural landmarks. Include realistic lat/lng for every location."
    elif tool_name == "get_travel_tips":
        city = tool_input.get("city", "")
        return f"Tips for {city}: use public metro/bus, carry local cash for markets, book popular spots in advance, learn basic local phrases, respect local customs, central neighbourhoods best for access."
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

def get_travel_plan(destination, trip_length, month, dietary):
    if "api_history" not in st.session_state:
        st.session_state.api_history = []
    dietary_str = f" Dietary restrictions: {', '.join(dietary)}." if dietary else ""
    st.session_state.api_history.append({
        "role": "user",
        "content": f"Plan me a {trip_length} day trip to {destination} in {month}.{dietary_str}"
    })
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
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
        st.session_state.api_history.append({"role": "user", "content": tool_results})
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=st.session_state.api_history
        )
    final_text = response.content[0].text
    st.session_state.api_history.append({"role": "assistant", "content": final_text})
    start = final_text.find("{")
    end = final_text.rfind("}") + 1
    if start == -1:
        raise ValueError("No JSON found in response")
    plan = json.loads(final_text[start:end])
    return plan, tools_used

def build_day_map(day):
    all_points = []
    for a in day.get("attractions", []):
        if a.get("lat"): all_points.append((a["lat"], a["lng"]))
    for f in day.get("food", []):
        if f.get("lat"): all_points.append((f["lat"], f["lng"]))
    for s in day.get("stay", []):
        if s.get("lat"): all_points.append((s["lat"], s["lng"]))
    if not all_points:
        return None
    center_lat = sum(p[0] for p in all_points) / len(all_points)
    center_lng = sum(p[1] for p in all_points) / len(all_points)
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13, tiles="OpenStreetMap")
    for idx, a in enumerate(day.get("attractions", [])):
        if a.get("lat") and a.get("lng"):
            folium.Marker([a["lat"], a["lng"]],
                popup=folium.Popup(f"<b>{a.get('emoji','')} {a['name']}</b><br>🕐 {a.get('time')} · 📍 {a.get('area')}<br><i>{a.get('description','')}</i>", max_width=200),
                tooltip=f"🗺️ {idx+1}. {a['name']}",
                icon=folium.Icon(color="purple", icon="star")).add_to(m)
    for f in day.get("food", []):
        if f.get("lat") and f.get("lng"):
            folium.Marker([f["lat"], f["lng"]],
                popup=folium.Popup(f"<b>🍽️ {f['name']}</b><br>{f.get('meal')} · {f.get('description','')}", max_width=200),
                tooltip=f"🍽️ {f['name']}",
                icon=folium.Icon(color="orange", icon="cutlery")).add_to(m)
    for s in day.get("stay", []):
        if s.get("lat") and s.get("lng"):
            folium.Marker([s["lat"], s["lng"]],
                popup=folium.Popup(f"<b>🏨 {s['name']}</b><br>⭐ {s.get('tier')} · 📍 {s.get('area')}", max_width=200),
                tooltip=f"🏨 {s['name']}",
                icon=folium.Icon(color="green", icon="home")).add_to(m)
    route_points = [[a["lat"], a["lng"]] for a in day.get("attractions", []) if a.get("lat")]
    if len(route_points) > 1:
        folium.PolyLine(route_points, color="#534AB7", weight=2.5, opacity=0.7, dash_array="6").add_to(m)
    legend = """<div style="position:fixed;bottom:20px;left:20px;z-index:1000;background:white;padding:8px 12px;border-radius:8px;border:1px solid #ccc;font-size:12px">
    🟣 Attractions &nbsp; 🟠 Food &nbsp; 🟢 Hotels</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m

def build_overview_map(plan):
    days = plan.get("days", [])
    airport = plan.get("airport", {})
    day_colors = ["red", "blue", "darkgreen", "purple", "orange", "darkred", "cadetblue"]
    all_points = []
    for day in days:
        for a in day.get("attractions", []):
            if a.get("lat"): all_points.append((a["lat"], a["lng"]))
    if airport.get("lat"): all_points.append((airport["lat"], airport["lng"]))
    if not all_points:
        return None
    center_lat = sum(p[0] for p in all_points) / len(all_points)
    center_lng = sum(p[1] for p in all_points) / len(all_points)
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12, tiles="OpenStreetMap")
    if airport.get("lat") and airport.get("lng"):
        folium.Marker([airport["lat"], airport["lng"]],
            popup=folium.Popup(f"<b>✈️ {airport.get('name','Airport')}</b><br>{airport.get('code','')}", max_width=200),
            tooltip=f"✈️ {airport.get('name','Airport')}",
            icon=folium.Icon(color="black", icon="plane")).add_to(m)
    for day in days:
        day_num = day["day"]
        color = day_colors[(day_num - 1) % len(day_colors)]
        day_points = []
        for a in day.get("attractions", []):
            if a.get("lat") and a.get("lng"):
                day_points.append([a["lat"], a["lng"]])
                folium.Marker([a["lat"], a["lng"]],
                    popup=folium.Popup(f"<b>Day {day_num}: {a.get('emoji','')} {a['name']}</b><br>📍 {a.get('area')}", max_width=200),
                    tooltip=f"D{day_num}: {a['name']}",
                    icon=folium.Icon(color=color, icon="info-sign")).add_to(m)
        for f in day.get("food", []):
            if f.get("lat") and f.get("lng"):
                folium.CircleMarker([f["lat"], f["lng"]], radius=6, color="orange", fill=True,
                    fill_color="orange", fill_opacity=0.8,
                    popup=folium.Popup(f"<b>🍽️ {f['name']}</b><br>{f.get('meal')} — Day {day_num}", max_width=150),
                    tooltip=f"🍽️ {f['name']}").add_to(m)
        for s in day.get("stay", []):
            if s.get("lat") and s.get("lng"):
                folium.CircleMarker([s["lat"], s["lng"]], radius=7, color="green", fill=True,
                    fill_color="green", fill_opacity=0.8,
                    popup=folium.Popup(f"<b>🏨 {s['name']}</b><br>⭐ {s.get('tier')} — Day {day_num}", max_width=150),
                    tooltip=f"🏨 {s['name']}").add_to(m)
        if len(day_points) > 1:
            folium.PolyLine(day_points, color=color, weight=3, opacity=0.8, dash_array="5",
                tooltip=f"Day {day_num} route").add_to(m)
    day1_attractions = [a for a in days[0].get("attractions", []) if a.get("lat")]
    if airport.get("lat") and day1_attractions:
        folium.PolyLine([[airport["lat"], airport["lng"]], [day1_attractions[0]["lat"], day1_attractions[0]["lng"]]],
            color="black", weight=2, opacity=0.5, dash_array="10", tooltip="Airport → Day 1").add_to(m)
    legend_html = """<div style="position:fixed;bottom:20px;left:20px;z-index:1000;background:white;padding:10px 14px;border-radius:8px;border:1px solid #ccc;font-size:12px;line-height:1.8">
    <b>Full Trip Map</b><br>✈️ Airport &nbsp;|&nbsp; 🔴 Day 1 &nbsp;|&nbsp; 🔵 Day 2 &nbsp;|&nbsp; 🟢 Day 3<br>🟠 Restaurants &nbsp;|&nbsp; 🟢 Hotels</div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

def speech_button(text, lang_code, key):
    """Renders a speak button using Web Speech API"""
    safe_text = text.replace("'", "\\'").replace('"', '\\"')
    html = f"""
    <button onclick="
        var u = new SpeechSynthesisUtterance('{safe_text}');
        u.lang = '{lang_code}';
        u.rate = 0.8;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
    " style="
        background: #EEEDFE;
        border: 1px solid #534AB7;
        border-radius: 8px;
        padding: 4px 10px;
        cursor: pointer;
        font-size: 14px;
        color: #534AB7;
    ">🔊 Hear it</button>
    """
    st.components.v1.html(html, height=40)


# ── Streamlit UI ───────────────────────────────────────────────
st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")

if "api_history" not in st.session_state:
    st.session_state.api_history = []
if "plan" not in st.session_state:
    st.session_state.plan = None
if "tools_used" not in st.session_state:
    st.session_state.tools_used = []
if "packing_quantities" not in st.session_state:
    st.session_state.packing_quantities = {}

with st.sidebar:
    st.title("✈️ AI Travel Planner")
    st.caption("Powered by Claude")
    st.divider()
    destination = st.text_input("🌍 Destination", placeholder="e.g. Tokyo, Paris, Bali")
    month = st.selectbox("📅 Month of travel", [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ])
    trip_length = st.slider("🗓️ Trip length (days)", 1, 7, 3)
    st.markdown("**🥗 Dietary restrictions**")
    dietary = []
    if st.checkbox("Vegetarian"): dietary.append("Vegetarian")
    if st.checkbox("Vegan"): dietary.append("Vegan")
    if st.checkbox("Halal"): dietary.append("Halal")
    if st.checkbox("Gluten-free"): dietary.append("Gluten-free")
    if st.checkbox("No seafood"): dietary.append("No seafood")
    if st.checkbox("No nuts"): dietary.append("No nuts")
    st.divider()
    if st.button("🚀 Plan my trip", use_container_width=True):
        if destination:
            st.session_state.api_history = []
            st.session_state.plan = None
            st.session_state.tools_used = []
            st.session_state.packing_quantities = {}
            with st.spinner("Planning your trip..."):
                try:
                    plan, tools_used = get_travel_plan(destination, trip_length, month, dietary)
                    st.session_state.plan = plan
                    st.session_state.tools_used = tools_used
                    for item in plan.get("packing_list", {}).get("clothing", []):
                        key = f"pack_{item['item']}"
                        st.session_state.packing_quantities[key] = item["quantity"]
                except Exception as e:
                    error_msg = str(e)
                    if "Expecting" in error_msg or "JSON" in error_msg:
                        st.error("Response too long — try reducing trip length to 2-3 days and try again.")
                    else:
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
    for item in ["🌤️ Real weather API", "🗺️ Google Places", "✈️ Flight search", "🏨 Hotel booking", "💱 Live exchange rates"]:
        st.caption(item)
    if st.session_state.plan:
        st.divider()
        if st.button("🗑️ Clear plan", use_container_width=True):
            st.session_state.api_history = []
            st.session_state.plan = None
            st.session_state.tools_used = []
            st.session_state.packing_quantities = {}
            st.rerun()

st.title("✈️ AI Travel Planner")

if not st.session_state.plan:
    st.markdown("""
    ### 👋 Welcome!
    **How it works:**
    1. Enter your destination in the sidebar
    2. Pick month, trip length, and any dietary needs
    3. Hit **Plan my trip** 🚀
    4. Get a full itinerary with maps, food, language cards with voice, and a packing list!

    **Powered by 3 AI tools:**
    - 🌤️ Weather checker · 🗺️ Attractions finder · 💡 Local travel tips
    """)
else:
    plan = st.session_state.plan
    lang_code = plan.get("language_code", "en-US")

    st.subheader(f"📍 {plan.get('destination')}, {plan.get('country','')} · {plan.get('duration')} days · {plan.get('month')}")
    st.info(f"🌤️ {plan.get('weather_summary','')}")

    days = plan.get("days", [])

    # Summary tab first, then day tabs
    tab_labels = ["📋 Summary"] + [f"Day {d['day']} — {d['theme']}" for d in days]
    tabs = st.tabs(tab_labels)

    # ── Summary tab (first) ────────────────────────────────────
    with tabs[0]:

        # 1. Full trip overview map
        st.markdown("### 🗺️ Full trip overview")
        st.caption("All days color coded · ✈️ Airport · 🟠 Restaurants · 🟢 Hotels")
        overview = build_overview_map(plan)
        if overview:
            st_folium(overview, width=None, height=450, returned_objects=[])
        else:
            st.caption("Overview map unavailable.")

        st.divider()

        # 2. Where to stay summary
        st.markdown("### 🏨 Where to stay")
        st.caption("All accommodation across your trip")
        for day in days:
            st.markdown(f"**Day {day['day']} — {day['theme']}**")
            hotel_cols = st.columns(max(len(day.get("stay", [])), 1))
            for idx, s in enumerate(day.get("stay", [])):
                with hotel_cols[idx]:
                    tier_color = {"Budget": "#4CAF50", "Mid-range": "#2196F3", "Luxury": "#9C27B0"}.get(s.get("tier"), "#888")
                    st.markdown(f"""
<div style="border:1px solid #e0e0e0;border-radius:10px;padding:12px;margin-bottom:8px;text-align:center">
<div style="font-size:11px;font-weight:600;color:{tier_color};margin-bottom:4px">{s.get('tier','').upper()}</div>
<b>{s.get('name')}</b><br>
<small>📍 {s.get('area')}</small>
</div>""", unsafe_allow_html=True)
                    if st.button(f"🔗 Book now", key=f"book_{day['day']}_{idx}", use_container_width=True):
                        st.info(f"Coming soon! Search **{s.get('name')}** on Booking.com or Hotels.com")

        st.divider()

        # 3. Language cards with voice + live translator
        st.markdown("### 🌐 Language & Translation")

        # Live translator
        st.markdown("##### 🔤 Translate anything")
        st.caption(f"Type any English phrase and get it in {plan.get('destination')} language instantly")
        t_col1, t_col2 = st.columns([3, 1])
        with t_col1:
            user_phrase = st.text_input("Type a phrase in English", placeholder="e.g. Can I have the bill please?", label_visibility="collapsed")
        with t_col2:
            translate_clicked = st.button("🌐 Translate", use_container_width=True)

        if translate_clicked and user_phrase:
            with st.spinner("Translating..."):
                translate_response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=300,
                    messages=[{
                        "role": "user",
                        "content": f"""Translate this English phrase to the local language of {plan.get('destination')}, {plan.get('country')}.

Phrase: "{user_phrase}"

Respond with ONLY a JSON object, no explanation:
{{"local": "translation here", "phonetic": "pronunciation guide here", "tip": "optional cultural tip or null"}}"""
                    }]
                )
                try:
                    raw = translate_response.content[0].text
                    start = raw.find("{")
                    end = raw.rfind("}") + 1
                    result = json.loads(raw[start:end])
                    st.markdown(f"""
<div style="border:2px solid #534AB7;border-radius:12px;padding:16px;margin:8px 0;background:#EEEDFE">
<div style="font-size:12px;color:#7F77DD;margin-bottom:4px">🇬🇧 {user_phrase}</div>
<div style="font-size:24px;font-weight:600;color:#26215C;margin:6px 0">{result.get('local', '')}</div>
<div style="font-size:13px;color:#534AB7;font-style:italic">/{result.get('phonetic', '')}/</div>
{"<div style='font-size:12px;color:#3C3489;margin-top:8px'>💡 " + result.get('tip') + "</div>" if result.get('tip') else ""}
</div>""", unsafe_allow_html=True)
                    speech_button(result.get("local", ""), lang_code, key="live_translation_speech")
                except Exception:
                    st.error("Translation failed — try again")

        st.divider()

        # Common phrase cards
        st.markdown("##### 📋 Common phrases")
        st.caption(f"Tap 🔊 to hear the pronunciation")
        lang_cards = plan.get("language_cards", [])
        lang_cols = st.columns(2)
        for idx, card in enumerate(lang_cards):
            with lang_cols[idx % 2]:
                st.markdown(f"""
<div style="border:1px solid #e0e0e0;border-radius:10px;padding:12px;margin-bottom:4px">
<div style="font-size:12px;color:gray">🇬🇧 {card.get('english')}</div>
<div style="font-size:20px;font-weight:600;margin:4px 0">{card.get('local')}</div>
<div style="font-size:12px;color:#7F77DD;font-style:italic">/{card.get('phonetic')}/</div>
</div>""", unsafe_allow_html=True)
                speech_button(card.get("local", ""), lang_code, key=f"speech_{idx}")

        st.divider()

        # 4. Currency
        st.markdown("### 💱 Currency & Budget")
        currency = plan.get("currency", {})
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Local Currency", currency.get("local_currency", "—"))
        with c2:
            st.metric("Daily Budget", currency.get("daily_budget_estimate", "—"))
        with c3:
            st.metric("Payment", currency.get("cash_or_card", "—").split(" ")[0])
        st.markdown(f"💡 **Exchange tip:** {currency.get('exchange_tip','')}")
        st.markdown(f"💰 **Useful amounts:** {currency.get('useful_amounts','')}")

        st.divider()

        # 5. Packing list
        st.markdown("### 🧳 Packing list")
        packing = plan.get("packing_list", {})
        pack_col1, pack_col2 = st.columns(2)
        with pack_col1:
            st.markdown("##### 👕 Clothing")
            st.caption("Adjust quantities to your needs")
            for item in packing.get("clothing", []):
                key = f"pack_{item['item']}"
                if key not in st.session_state.packing_quantities:
                    st.session_state.packing_quantities[key] = item["quantity"]
                ca, cb = st.columns([3, 1])
                with ca:
                    st.markdown(f"<div style='padding:6px 0'>{item['item']}</div>", unsafe_allow_html=True)
                with cb:
                    st.session_state.packing_quantities[key] = st.number_input(
                        "qty", min_value=0, max_value=20,
                        value=st.session_state.packing_quantities[key],
                        key=f"input_{key}", label_visibility="collapsed"
                    )
        with pack_col2:
            st.markdown("##### 🎒 Essentials checklist")
            essentials = packing.get("essentials", [])
            for idx, item in enumerate(essentials):
                st.checkbox(item, key=f"essential_{idx}")
            packed = sum(1 for idx in range(len(essentials)) if st.session_state.get(f"essential_{idx}", False))
            if essentials:
                st.progress(packed / len(essentials), text=f"{packed} of {len(essentials)} essentials packed")

        st.divider()

        # 6. Next steps
        st.markdown("### ✅ Next steps")
        st.caption("Check these off as you prepare!")
        next_steps = plan.get("next_steps", [])
        ns_cols = st.columns(2)
        for idx, step in enumerate(next_steps):
            with ns_cols[idx % 2]:
                st.checkbox(step, key=f"step_{idx}")
        done = sum(1 for idx in range(len(next_steps)) if st.session_state.get(f"step_{idx}", False))
        if next_steps:
            st.progress(done / len(next_steps), text=f"{done} of {len(next_steps)} steps done")

    # ── Day tabs ───────────────────────────────────────────────
    for i, tab in enumerate(tabs[1:]):
        with tab:
            day = days[i]
            st.markdown("#### 🗺️ Day map")
            st.caption("🟣 Attractions &nbsp;|&nbsp; 🟠 Food &nbsp;|&nbsp; 🟢 Hotels")
            m = build_day_map(day)
            if m:
                st_folium(m, width=None, height=380, returned_objects=[])
            else:
                st.caption("Map unavailable for this day.")
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("##### 🗺️ Attractions")
                for idx, a in enumerate(day.get("attractions", [])):
                    st.markdown(f"""
<div style="border:1px solid #e0e0e0;border-radius:10px;padding:10px;margin-bottom:8px">
<b>{idx+1}. {a.get('emoji','')} {a.get('name')}</b><br>
<small>🕐 {a.get('time')} · 📍 {a.get('area')}</small><br>
<small style="color:gray">{a.get('description')}</small>
</div>""", unsafe_allow_html=True)
            with col2:
                st.markdown("##### 🍽️ Where to eat")
                for f in day.get("food", []):
                    st.markdown(f"""
<div style="border:1px solid #e0e0e0;border-radius:10px;padding:10px;margin-bottom:8px">
<b>{f.get('name')}</b><br>
<small>🍴 {f.get('meal')}</small><br>
<small style="color:gray">{f.get('description')}</small>
</div>""", unsafe_allow_html=True)
                st.markdown("##### 🏨 Where to stay")
                for s in day.get("stay", []):
                    st.markdown(f"""
<div style="border:1px solid #e0e0e0;border-radius:10px;padding:10px;margin-bottom:8px">
<b>{s.get('name')}</b><br>
<small>⭐ {s.get('tier')} · 📍 {s.get('area')}</small>
</div>""", unsafe_allow_html=True)
            with col3:
                st.markdown("##### 👗 What to wear")
                for item in plan.get("what_to_wear", []):
                    st.markdown(f"- {item}")
                st.markdown("##### 💡 Travel tips")
                tips = plan.get("travel_tips", [])
                if tips:
                    st.info(tips[i % len(tips)])