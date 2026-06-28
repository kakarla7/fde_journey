import anthropic
import os
import re
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── Step 1: Load document ──────────────────────────────────────
def load_document(filepath):
    with open(filepath, "r") as f:
        return f.read()

# ── Step 2: Split into chunks ──────────────────────────────────
def chunk_document(text, chunk_size=500, overlap=50):
    """
    Split document into overlapping chunks of ~chunk_size words.
    Overlap means chunks share some words at the boundary
    so we don't lose context at the edges.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append({
            "id": len(chunks),
            "text": chunk,
            "word_start": start,
            "word_end": end
        })
        # Move forward by chunk_size minus overlap
        start += chunk_size - overlap

    return chunks

# ── Step 3: Find relevant chunks ───────────────────────────────
def find_relevant_chunks(question, chunks, top_k=3):
    """
    Simple keyword search — score each chunk by how many
    question words appear in it. Return top_k chunks.
    """
    # Clean question into keywords
    stop_words = {"what", "is", "the", "a", "an", "how", "why", "when",
                  "where", "who", "are", "does", "do", "in", "of", "to",
                  "and", "or", "can", "tell", "me", "about", "explain"}

    question_words = set(
        re.sub(r'[^\w\s]', '', question.lower()).split()
    ) - stop_words

    scored = []
    for chunk in chunks:
        chunk_text_lower = chunk["text"].lower()
        # Count how many question keywords appear in this chunk
        score = sum(1 for word in question_words if word in chunk_text_lower)
        if score > 0:
            scored.append((score, chunk))

    # Sort by score descending, return top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k]]

# ── Step 4: Ask Claude with only relevant chunks ───────────────
def ask_with_chunks(question, relevant_chunks, conversation_history):
    if not relevant_chunks:
        return "I couldn't find relevant information in the document for that question.", conversation_history

    # Build context from relevant chunks only
    context = "\n\n---\n\n".join([
        f"[Chunk {c['id']}]\n{c['text']}"
        for c in relevant_chunks
    ])

    if not conversation_history:
        # First question
        conversation_history.append({
            "role": "user",
            "content": f"""Answer the question using only the document chunks below.
Always mention which chunk your answer came from.
If the answer isn't in the chunks, say so.

<document_chunks>
{context}
</document_chunks>

Question: {question}"""
        })
    else:
        # Follow-up — include fresh chunks each time
        conversation_history.append({
            "role": "user",
            "content": f"""Here are relevant chunks for this follow-up question:

<document_chunks>
{context}
</document_chunks>

Question: {question}"""
        })

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="You are a helpful assistant that answers questions based strictly on provided document chunks. Always cite which chunk number your answer comes from.",
        messages=conversation_history
    )

    answer = response.content[0].text
    conversation_history.append({
        "role": "assistant",
        "content": answer
    })

    return answer, conversation_history

# ── Main ───────────────────────────────────────────────────────
def main():
    doc_path = "week-01/projects/sample_doc.txt"
    document = load_document(doc_path)

    # Chunk it
    chunks = chunk_document(document, chunk_size=150, overlap=20)

    print(f"✅ Document loaded — {len(document.split())} words")
    print(f"✅ Split into {len(chunks)} chunks (~150 words each)")
    print(f"\n📄 Chat with your document. Type 'quit' to exit.\n")

    conversation_history = []

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        # Find relevant chunks
        relevant = find_relevant_chunks(question, chunks, top_k=3)
        print(f"🔍 Found {len(relevant)} relevant chunks: {[c['id'] for c in relevant]}")

        # Ask Claude
        answer, conversation_history = ask_with_chunks(
            question, relevant, conversation_history
        )
        print(f"\nAssistant: {answer}\n")

if __name__ == "__main__":
    main()