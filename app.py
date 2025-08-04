from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from sentence_transformers import SentenceTransformer
import chromadb
import streamlit as st

load_dotenv()
groq_model = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key = groq_model,
    model_name = "gemma2-9b-it"
)

help_doc = "./help_docs"
vector_db = "./chroma_db"

embedder = SentenceTransformer("all-MiniLM-L6-v2") 
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("help_docs")


memory = MemorySaver()

def help_docs_embed():
    if not os.path.exists(help_doc):
        st.error(f"Help docs directory '{help_doc}' not found!")
        return

    for filename in os.listdir(help_doc):
        if filename.endswith(".txt"):
            filepath = os.path.join(help_doc, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if content and filename not in set(collection.get()["ids"]):
                embedding = embedder.encode(content).tolist()
                collection.add(
                    documents=[content],
                    ids=[filename],
                    embeddings=[embedding]
                )

class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str

def retrieve_relevant_docs(query: str, k=3):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding], n_results=k
    )

    print("Retrieved documents for query:")
    for doc in results["documents"][0]:
        print("Document " , doc[:1]) 

    return results["documents"][0] if results["documents"] else []
    

def chatbot(state : State) -> State:
    last_message = state["messages"][-1].content
    relevant_docs = retrieve_relevant_docs(last_message)

    context = "\n".join(relevant_docs)
    enhanced_prompt = f"Context:\n{context}\n\nQuestion: {last_message}"

    messages = state["messages"][:-1] + [HumanMessage(content=enhanced_prompt)]
    response = llm.invoke(messages)

    return {"messages": [response], "context": context}


builder = StateGraph(State)
builder.add_node("chatbot_node", chatbot)
builder.add_edge(START, "chatbot_node")
builder.add_edge("chatbot_node", END)
graph = builder.compile(checkpointer=memory)

help_docs_embed()


st.set_page_config(page_title="EduHelp", page_icon=":books:")
st.header("Chat with EduHelp :books:")


if "messages" not in st.session_state:
    st.session_state["messages"] = []


def convert_to_lc_messages(history):
    lc_msgs = []
    for msg in history:
        if msg["role"] == "user":
            lc_msgs.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_msgs.append(AIMessage(content=msg["content"]))
    return lc_msgs
user_input = st.chat_input("Ask...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.spinner("Processing..."):
        config = {'configurable': {'thread_id': '1'}}

        last_3_exchanges = st.session_state["messages"][-6:]
        lc_messages = convert_to_lc_messages(last_3_exchanges)
        response = graph.invoke({"messages": lc_messages}, config=config)
        # with st.expander("Retrieved Context"):
        #     st.code(response["context"])
        
        chat_response = response["messages"][-1].content
        st.session_state.messages.append({"role": "assistant", "content": chat_response})

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"**User:** {msg['content']}")
    else:
        st.markdown(f"**EduHelp:** {msg['content']}")