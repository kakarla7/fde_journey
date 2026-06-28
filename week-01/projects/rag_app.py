import streamlit as st
import anthropic
import os
import re
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

# ── RAG functions ──────────────────────────────────────────────
def chunk_document(text, chunk_size=150, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append({"id": len(chunks), "text": chunk})
        start += chunk_size - overlap
    return chunks

def find_relevant_chunks(question, chunks, top_k=3):
    stop_words = {"what", "is", "the", "a", "an", "how", "why", "when",
                  "where", "who", "are", "does", "do", "in", "of", "to",
                  "and", "or", "can", "tell", "me", "about", "explain",
                  "give", "list", "describe", "define", "show"}
    question_words = set(re.sub(r'[^\w\s]', '', question.lower()).split()) - stop_words
    scored = []
    for chunk in chunks:
        chunk_lower = chunk["text"].lower()
        score = sum(1 for word in question_words if word in chunk_lower)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k]]

def ask_with_chunks(question, relevant_chunks, conversation_history):
    if not relevant_chunks:
        return "I couldn't find relevant information in the document for that question.", conversation_history

    context = "\n\n---\n\n".join([
        f"[Chunk {c['id']}]\n{c['text']}"
        for c in relevant_chunks
    ])

    if not conversation_history:
        conversation_history.append({
            "role": "user",
            "content": f"""Answer the question using only the document chunks below.
Always mention which chunk your answer came from.
If the answer isn't in the chunks, say so clearly.

<document_chunks>
{context}
</document_chunks>

Question: {question}"""
        })
    else:
        conversation_history.append({
            "role": "user",
            "content": f"""Here are the relevant chunks for this question:

<document_chunks>
{context}
</document_chunks>

Question: {question}"""
        })

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="You are a helpful assistant that answers questions strictly based on provided document chunks. Always cite the chunk number your answer comes from. Be concise and clear.",
        messages=conversation_history
    )

    answer = response.content[0].text
    conversation_history.append({"role": "assistant", "content": answer})
    return answer, conversation_history


# ── Streamlit UI ───────────────────────────────────────────────
st.set_page_config(page_title="Chat with your Doc", page_icon="📄", layout="wide")

# ── Session state ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_history" not in st.session_state:
    st.session_state.api_history = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "doc_text" not in st.session_state:
    st.session_state.doc_text = None

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Doc Chat")
    st.caption("Powered by Claude + RAG")
    st.divider()

    # File uploader
    st.markdown("**Upload your document**")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "md"],
        label_visibility="collapsed"
    )

    # Or use sample doc
    st.caption("or")
    use_sample = st.button("📄 Use sample doc", use_container_width=True)

    # Process uploaded file
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        st.session_state.doc_text = text
        st.session_state.doc_name = uploaded_file.name
        st.session_state.chunks = chunk_document(text)
        st.session_state.messages = []
        st.session_state.api_history = []
        st.rerun()

    # Process sample doc
    if use_sample:
        sample_path = "week-01/projects/sample_doc.txt"
        try:
            with open(sample_path, "r") as f:
                text = f.read()
            st.session_state.doc_text = text
            st.session_state.doc_name = "sample_doc.txt (AI Agents Guide)"
            st.session_state.chunks = chunk_document(text)
            st.session_state.messages = []
            st.session_state.api_history = []
            st.rerun()
        except FileNotFoundError:
            st.error("Sample doc not found. Make sure sample_doc.txt is in week-01/projects/")

    # Doc stats
    if st.session_state.doc_name:
        st.divider()
        st.markdown("**📊 Document stats**")
        st.success(f"✅ {st.session_state.doc_name}")
        word_count = len(st.session_state.doc_text.split())
        st.metric("Words", f"{word_count:,}")
        st.metric("Chunks", len(st.session_state.chunks))
        st.metric("Chunk size", "~150 words")

        st.divider()
        st.markdown("**💡 Try asking:**")
        sample_questions = [
            "What is the agent loop?",
            "What types of memory do agents have?",
            "What skills does an FDE need?",
            "What is RAG?",
            "What are the challenges in agent development?"
        ]
        for q in sample_questions:
            if st.button(q, use_container_width=True, key=f"sq_{q[:20]}"):
                st.session_state.pending_question = q

        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.api_history = []
            st.rerun()

# ── Main area ──────────────────────────────────────────────────
st.title("📄 Chat with your Document")
st.caption("Upload a document and ask questions — answers are grounded in your content")

if not st.session_state.doc_name:
    # Empty state
    st.markdown("""
    ### 👋 Get started
    **Upload a document in the sidebar** or use the sample doc.

    Supported formats: `.txt`, `.md`

    ---

    ### How it works
    1. **Upload** your document
    2. It gets **split into chunks**
    3. When you ask a question, the most **relevant chunks** are found
    4. Claude answers using **only those chunks**
    5. Every answer includes **which chunk** it came from

    ### What is RAG?
    RAG = **R**etrieval **A**ugmented **G**eneration

    Instead of Claude guessing from training data, it reads YOUR document and answers from that.
    Perfect for private documents, recent reports, or anything Claude wasn't trained on.
    """)
else:
    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "chunks_used" in message:
                with st.expander(f"🔍 {len(message['chunks_used'])} chunks searched"):
                    for chunk in message["chunks_used"]:
                        st.markdown(f"""
<div style="background:#f8f9fa;border-left:3px solid #534AB7;padding:8px 12px;margin:4px 0;border-radius:4px;font-size:12px">
<b>Chunk {chunk['id']}</b><br>{chunk['text'][:200]}...
</div>""", unsafe_allow_html=True)

    # Handle sidebar question buttons
    if "pending_question" in st.session_state:
        prompt = st.session_state.pending_question
        del st.session_state.pending_question
    else:
        prompt = None

    # Chat input
    user_input = st.chat_input("Ask anything about your document...")
    if user_input:
        prompt = user_input

    if prompt:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Find relevant chunks
        relevant = find_relevant_chunks(prompt, st.session_state.chunks, top_k=3)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Reading document..."):
                answer, st.session_state.api_history = ask_with_chunks(
                    prompt, relevant, st.session_state.api_history
                )
            st.markdown(answer)

            # Show which chunks were used
            if relevant:
                with st.expander(f"🔍 {len(relevant)} chunks searched — click to see"):
                    for chunk in relevant:
                        st.markdown(f"""
<div style="background:#f8f9fa;border-left:3px solid #534AB7;padding:8px 12px;margin:4px 0;border-radius:4px;font-size:12px">
<b>Chunk {chunk['id']}</b><br>{chunk['text'][:200]}...
</div>""", unsafe_allow_html=True)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "chunks_used": relevant
        })