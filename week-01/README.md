# Week 1 Showcase — Claude API Foundations

## 🎯 What I built
FDE Coach — a live AI chatbot powered by Claude API, 
deployed publicly on Streamlit Cloud.

## ✅ What I learned
- How to call the Anthropic Claude API from Python
- System prompts and how to give Claude a persona
- Tool use — giving Claude the ability to call functions
- Parallel tool use — handling multiple tools in one response
- Multi-turn conversation history management
- Deploying a Python app to Streamlit Cloud

## 💻 Key concepts
- `api_history` vs display `messages` — why they're separate
- Serializing SDK objects to plain dicts before sending back to API
- Using a `while` loop for tool handling, not `if`

## 🔗 Live demo
[FDE Coach — Live App](https://fdejourney.streamlit.app)
Access code: ..........

## 🧠 Biggest lesson
Claude has no memory by default — you have to send the 
entire conversation history every single API call.
Tool use requires exact pairing of tool_use blocks with 
tool_result blocks in the very next message.

## ➡️ Next week
Hugging Face AI Agents Course — understanding agent loops,
memory, and open source models.