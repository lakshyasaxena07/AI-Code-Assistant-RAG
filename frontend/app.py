import streamlit as st
import requests
import json
import websocket
from dotenv import load_dotenv
import os

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
WS_URL = os.getenv("WS_URL", "ws://localhost:8000")

st.set_page_config(page_title="AI Code Assistant", page_icon="💻", layout="wide")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "repos" not in st.session_state:
    st.session_state.repos = []
if "selected_repo" not in st.session_state:
    st.session_state.selected_repo = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# --- Helper Functions ---
def fetch_repos():
    try:
        res = requests.get(f"{BACKEND_URL}/api/ingest/repos")
        if res.status_code == 200:
            st.session_state.repos = res.json().get("repos", [])
    except Exception as e:
        st.error(f"Failed to fetch repos: {e}")

# --- Sidebar ---
with st.sidebar:
    st.title("💻 Code Assistant")
    
    st.header("1. Ingest Repository")
    github_url = st.text_input("GitHub URL")
    if st.button("Ingest", type="primary"):
        if not github_url:
            st.warning("Please enter a URL")
        else:
            with st.spinner("Ingesting repository... This may take a minute."):
                try:
                    res = requests.post(
                        f"{BACKEND_URL}/api/ingest/github",
                        json={"github_url": github_url}
                    )
                    if res.status_code == 200:
                        st.success("Successfully ingested!")
                        fetch_repos()
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    
    st.header("2. Select Repository")
    if st.button("Refresh Repos"):
        fetch_repos()
        
    if not st.session_state.repos:
        fetch_repos()
        
    repo_options = {repo["repo_name"]: repo["repo_id"] for repo in st.session_state.repos}
    
    if repo_options:
        # Prevent ValueError if selected_repo is no longer in options
        if st.session_state.selected_repo not in repo_options.values():
            st.session_state.selected_repo = list(repo_options.values())[0]
            
        selected_name = st.selectbox(
            "Active Repository",
            options=list(repo_options.keys()),
            index=list(repo_options.values()).index(st.session_state.selected_repo)
        )
        st.session_state.selected_repo = repo_options[selected_name]
        st.info(f"Using: {selected_name}")
    else:
        st.info("No repositories ingested yet.")
        
    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()

# --- Main Interface ---
st.title("Chat with your Codebase")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "retrieval_info" in msg:
            with st.expander("🔍 Retrieval Info"):
                st.json(msg["retrieval_info"])

# Chat Input
if prompt := st.chat_input("Ask about your code..."):
    if not st.session_state.selected_repo:
        st.error("Please select a repository first from the sidebar.")
        st.stop()
        
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        retrieval_info_placeholder = st.empty()
        
        full_response = ""
        retrieval_info = None
        
        # Connect to WebSocket
        try:
            ws = websocket.WebSocket()
            ws.connect(f"{WS_URL}/api/query/stream")
            
            payload = {
                "repo_id": st.session_state.selected_repo,
                "question": prompt,
                "conversation_id": st.session_state.conversation_id
            }
            ws.send(json.dumps(payload))
            
            while True:
                result = ws.recv()
                data = json.loads(result)
                
                msg_type = data.get("type")
                
                if msg_type == "token":
                    full_response += data["data"]
                    response_placeholder.markdown(full_response + "▌")
                elif msg_type == "retrieval_info":
                    retrieval_info = data["data"]
                    with retrieval_info_placeholder.expander("🔍 Retrieval Info"):
                        st.json(retrieval_info)
                elif msg_type == "done":
                    # Update conversation id
                    st.session_state.conversation_id = data["data"].get("conversation_id")
                    break
                elif msg_type == "error":
                    st.error(f"Error: {data['data']}")
                    break
            
            ws.close()
            response_placeholder.markdown(full_response)
            
            msg_obj = {"role": "assistant", "content": full_response}
            if retrieval_info:
                msg_obj["retrieval_info"] = retrieval_info
            st.session_state.messages.append(msg_obj)
            
        except Exception as e:
            st.error(f"Connection error: {e}")
