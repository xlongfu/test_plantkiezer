from __future__ import annotations

import pandas as pd
import streamlit as st

from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant


st.set_page_config(page_title="Plantkiezer.nl", page_icon="🌿", layout="wide")
st.title("🌿 Plantkiezer Plant Recommender")
st.caption("Type your plant needs; I'll recommend and show product cards.")

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
api_key = st.secrets["API_KEY"]



llm = init_chat_model(
    "accounts/fireworks/models/llama-v3p1-405b-instruct",
    model_provider="fireworks",
    api_key=api_key,
    model_kwargs={
        "max_tokens": 1024,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
    },
)

# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("What is up?"):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert session messages to langchain message objects
    lc_msgs = []
    for m in st.session_state.messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system":
            lc_msgs.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_msgs.append(AIMessage(content=content))
        else:
            lc_msgs.append(HumanMessage(content=content))
    
        embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vectorstore = Qdrant.from_existing_collection(
        collection_name="planten",
        embedding=embeddings,
        path="vector_stores/plantkiezer1",
    )

    query = st.session_state.messages[-1]["content"]


    retrieved_docs = list(vectorstore.max_marginal_relevance_search(query, k=3, filter=None))


    def format_docs(docs):
        blocks = []
        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("source") or d.metadata.get("id") or f"doc-{i}"
            blocks.append(f"[{i}] Source: {src}\n{d.page_content}")
        return "\n\n".join(blocks)

    context = format_docs(retrieved_docs)

    grounding = (
        "You are a helpful assistant. Use ONLY the context to answer.\n"
        "If the answer isn't in the context, say you don't know.\n"
        "Cite inline like [1], [2] based on the context blocks.\n\n"
        f"Context:\n{context}\n"
        "---"
    )

    # Prepend grounding before the running chat history,
    # then reinforce the latest user question explicitly.
    grounded_msgs = [SystemMessage(content=grounding)] + lc_msgs

    # Optionally, ensure the latest user question is crystal clear:
    # grounded_msgs.append(HumanMessage(content=f"My question: {query}"))

    # --- Generate using your Fireworks LLM via LangChain ---
    resp = llm.generate([grounded_msgs])
    assistant_text = ""
    try:
        assistant_text = resp.generations[0][0].text or ""
    except Exception:
        assistant_text = "Sorry, I couldn't generate a response."

    # --- Optional: add a sources footer from doc metadata ---
    def collect_citations(docs):
        lines = []
        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("source") or d.metadata.get("id") or f"doc-{i}"
            title = d.metadata.get("title")
            lines.append(f"[{i}] {title+' — ' if title else ''}{src}")
        return "\n".join(lines)

    if retrieved_docs:
        assistant_text += "\n\n---\nSources:\n" + collect_citations(retrieved_docs)

    # --- Display + store ---
    with st.chat_message("assistant"):
        st.markdown(assistant_text)
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
