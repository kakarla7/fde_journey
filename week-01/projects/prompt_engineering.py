import anthropic
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def call_claude(prompt, system=None, max_tokens=1024):
    kwargs = {
        "model": "claude-sonnet-4-6",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return response.content[0].text

print("=" * 60)
print("PROMPT ENGINEERING — 6 Core Techniques")
print("=" * 60)

# ── 1. Zero-shot ───────────────────────────────────────────────
print("\n1️⃣  ZERO-SHOT — just ask, no examples")
print("-" * 40)

result = call_claude("Classify this customer message as: positive, negative, or neutral.\n\nMessage: 'The product arrived late but works perfectly.'")
print(f"Result: {result}")

# ── 2. Few-shot ────────────────────────────────────────────────
print("\n2️⃣  FEW-SHOT — teach by example first")
print("-" * 40)

few_shot_prompt = """Classify customer messages as positive, negative, or neutral.

Examples:
Message: "Love this product, works perfectly!" → positive
Message: "Broken on arrival, terrible quality" → negative
Message: "Package arrived today" → neutral
Message: "Great value but shipping was slow" → positive

Now classify this:
Message: "Support team was unhelpful but product is fine" →"""

result = call_claude(few_shot_prompt)
print(f"Result: {result}")
print("\n💡 Notice: few-shot is more consistent than zero-shot")

# ── 3. Chain of thought ────────────────────────────────────────
print("\n3️⃣  CHAIN OF THOUGHT — make Claude think out loud")
print("-" * 40)

# Without CoT
simple = call_claude("A client has 500 employees. Each uses an AI tool for 2 hours/day. The tool costs $0.02 per hour per user. What's the monthly cost?")
print(f"Without CoT: {simple}")

# With CoT
cot_prompt = """A client has 500 employees. Each uses an AI tool for 2 hours/day. 
The tool costs $0.02 per hour per user. What's the monthly cost?

Think through this step by step before giving the final answer."""

result = call_claude(cot_prompt)
print(f"\nWith CoT:\n{result}")
print("\n💡 CoT reduces errors on math and logic problems")

# ── 4. Structured output ───────────────────────────────────────
print("\n4️⃣  STRUCTURED OUTPUT — force a specific format")
print("-" * 40)

structured_prompt = """Analyze this business problem and respond with ONLY a JSON object.
No explanation, no markdown, just the JSON.

Problem: "Our sales team spends 3 hours per day manually updating CRM records after calls."

Respond with:
{
  "problem_summary": "one sentence",
  "time_wasted_per_week_hours": number,
  "ai_solution": "one sentence",
  "estimated_time_saved_percent": number,
  "implementation_complexity": "low/medium/high",
  "recommended_tools": ["tool1", "tool2"]
}"""

result = call_claude(structured_prompt)
print(f"Raw result:\n{result}")

# Parse and use it
try:
    parsed = json.loads(result)
    print(f"\n✅ Parsed successfully!")
    print(f"Problem: {parsed['problem_summary']}")
    print(f"Time saved: {parsed['estimated_time_saved_percent']}%")
    print(f"Complexity: {parsed['implementation_complexity']}")
except:
    print("Parse failed — Claude added extra text")

print("\n💡 This is exactly how Travel Agent works — Claude returns JSON, code renders it")

# ── 5. Prompt chaining ─────────────────────────────────────────
print("\n5️⃣  PROMPT CHAINING — chain outputs together")
print("-" * 40)

client_problem = "We have 50 support agents answering the same 20 questions all day. It takes 5 minutes per answer."

# Chain step 1 — analyse
print("Step 1: Analysing problem...")
analysis = call_claude(f"""You are an FDE at an AI company.
Analyse this client problem in 2-3 sentences:
{client_problem}""")
print(f"Analysis: {analysis}")

# Chain step 2 — feed analysis into solution design
print("\nStep 2: Designing solution...")
solution = call_claude(f"""You are an FDE at an AI company.
Based on this analysis: {analysis}

Design a concrete AI agent solution in 3 bullet points.
Be specific about what the agent does, what tools it needs, and how it helps.""")
print(f"Solution: {solution}")

# Chain step 3 — feed solution into proposal
print("\nStep 3: Writing proposal...")
proposal = call_claude(f"""You are an FDE writing a one-paragraph client proposal.
Problem analysis: {analysis}
Proposed solution: {solution}

Write a compelling one paragraph proposal the client can approve today.
Be confident, specific, and outcome-focused.""")
print(f"\nProposal:\n{proposal}")

print("\n💡 This is prompt chaining — 3 focused prompts beats 1 complex prompt")

# ── 6. System prompt design ────────────────────────────────────
print("\n6️⃣  SYSTEM PROMPT DESIGN — persona + rules + constraints")
print("-" * 40)

same_question = "Should we use AI for our customer support?"

# Generic system prompt
generic = call_claude(
    same_question,
    system="You are a helpful assistant."
)
print(f"Generic assistant:\n{generic[:200]}...")

# FDE system prompt
fde_response = call_claude(
    same_question,
    system="""You are a Forward Deployed Engineer at Anthropic with 5 years experience 
deploying AI agents at Fortune 500 companies. 

Rules:
- Always give a direct recommendation first, then reasoning
- Use specific numbers and timeframes when possible  
- Reference real implementation patterns you've seen work
- Keep responses under 150 words
- End with one concrete next step the client can take today"""
)
print(f"\nFDE assistant:\n{fde_response}")
print("\n💡 Same question, completely different quality — system prompt is everything")

print("\n" + "=" * 60)
print("✅ All 6 techniques complete!")
print("=" * 60)