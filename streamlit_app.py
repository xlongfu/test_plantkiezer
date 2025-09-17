from __future__ import annotations

import pandas as pd
import streamlit as st

from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant


api_key = st.secrets["API_KEY"]


st.set_page_config(page_title="Plantkiezer.nl", page_icon="🌿", layout="wide")
st.title("🌿 Plantkiezer Plant Recommender")
st.caption("Type your plant needs; I'll recommend and show product cards.")

llm = init_chat_model(
    "accounts/fireworks/models/llama-v3p1-405b-instruct",
    model_provider="fireworks",
    api_key=api_key,
    model_kwargs={
        "max_tokens": 512,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.5,
    },
)

# init and show chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# create chat input field 
if prompt := st.chat_input("What kind of plants are you interested in?"):

    # store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # convert session messages to langchain message objects
    session_messages = []

    for m in st.session_state.messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system":
            session_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            session_messages.append(AIMessage(content=content))
        else:
            session_messages.append(HumanMessage(content=content))

    query = st.session_state.messages[-1]["content"]

    # ----- RAG process ----- 
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vectorstore = Qdrant.from_existing_collection(
        collection_name="planten",
        embedding=embeddings,
        path="vector_stores/plantkiezer1",
    )

    retrieved_docs = list(vectorstore.max_marginal_relevance_search(query, k=3, filter=None))


    def format_docs(docs):
        blocks = []

        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("source") or d.metadata.get("id") or f"doc-{i}"
            blocks.append(f"[{i}] Source: {src}\n{d.page_content}")
        
        return "\n\n".join(blocks)

    context = format_docs(retrieved_docs)

    instruction = (
        "You are an expert botanical assistant and also a sales chatbot. \n"
        "You will be provided with three retrieved plant entries. \n"
        f"Context:\n{context}\n\n"
        "Answer the user query by recommending these three plants. \n"
        "Use the descriptions of the retrieved data to also provide more \n"
        "information about the plants. Frame your response concisely, while \n"
        "also like a real salesperson. Here is the user question: \n\n"
    )

    all_context = [SystemMessage(content=instruction)] + session_messages

    # Optionally, ensure the latest user question is crystal clear:
    # grounded_msgs.append(HumanMessage(content=f"My question: {query}"))

    # --- Generate using your Fireworks LLM via LangChain ---
    response = llm.generate([all_context])
    assistant_text = ""

    try:
        assistant_text = response.generations[0][0].text or ""
    except Exception:
        assistant_text = "Sorry, I couldn't generate a response. Try again later."

    # --- Optional: add a sources footer from doc metadata ---
    # def collect_citations(docs):
    #     lines = []
    #     for i, d in enumerate(docs, start=1):
    #         src = d.metadata.get("source") or d.metadata.get("id") or f"doc-{i}"
    #         title = d.metadata.get("title")
    #         lines.append(f"[{i}] {title+' — ' if title else ''}{src}")
    #     return "\n".join(lines)

    # if retrieved_docs:
    #     assistant_text += "\n\n---\nSources:\n" + collect_citations(retrieved_docs)

    # --- Display + store ---
    with st.chat_message("assistant"):
        st.markdown(assistant_text)

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
