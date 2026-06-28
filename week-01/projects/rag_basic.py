import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def load_document(filepath):
    with open(filepath, "r") as f:
        return f.read()

def ask_document(document, question, conversation_history):
    # First message — include the document as context
    if not conversation_history:
        conversation_history.append({
            "role": "user",
            "content": f"""Here is a document I want you to answer questions about.
            
<document>
{document}
</document>

My first question is: {question}

Answer based only on what's in the document. If the answer isn't in the document, say so."""
        })
    else:
        # Follow-up questions — document already in history
        conversation_history.append({
            "role": "user",
            "content": question
        })

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="You are a helpful assistant that answers questions based strictly on provided documents. Always cite which section your answer comes from.",
        messages=conversation_history
    )

    answer = response.content[0].text
    conversation_history.append({
        "role": "assistant",
        "content": answer
    })

    return answer, conversation_history

def main():
    # Load the document
    doc_path = "week-01/projects/sample_doc.txt"
    document = load_document(doc_path)
    print(f"✅ Document loaded — {len(document)} characters, ~{len(document.split())} words")
    print("\n📄 Chat with your document. Type 'quit' to exit.\n")

    conversation_history = []

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        answer, conversation_history = ask_document(document, question, conversation_history)
        print(f"\nAssistant: {answer}\n")

if __name__ == "__main__":
    main()