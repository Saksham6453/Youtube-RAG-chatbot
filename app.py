import streamlit as st
import os
import time
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(
    page_title="YouTube AI", 
    page_icon="🎥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Custom CSS for Hero Section, Chat Bubbles, and Footer (Light Mode Optimized)
st.markdown("""
<style>
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #1E3A8A, #0EA5E9); /* Deep blue to bright blue */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: -10px;
        text-align: center;
    }
    .hero-subtitle {
        font-size: 1.3rem;
        color: #4B5563; /* Dark gray */
        text-align: center;
        margin-bottom: 40px;
    }
    .hero-tech {
        font-weight: 700;
        color: #111827; /* Almost black */
    }
    .footer {
        text-align: center;
        color: #6B7280; /* Medium gray */
        font-size: 0.9rem;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #E5E7EB; /* Light gray border */
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "video_url" not in st.session_state:
    st.session_state.video_url = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0
if "quick_question" not in st.session_state:
    st.session_state.quick_question = None

# ==========================================
# 2. Caching & Processing Logic
# ==========================================
@st.cache_resource(show_spinner=False)
def process_youtube_video(url: str):
    """Fetches, chunks, and embeds the YouTube video. Cached for performance."""
    # 1. Load
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False, language=["hi", "en"])
    docs = loader.load()
    
    # 2. Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # 3. Embed
    embeddings = NVIDIAEmbeddings(model="nvidia/llama-nemotron-embed-1b-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store, len(chunks)

# ==========================================
# 3. Sidebar & Workflow
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🎥 YouTube AI</h2>", unsafe_allow_html=True)
    st.markdown("━━━━━━━━━━━━━━━━━━━━━━")
    
    url = st.text_input("🔗 Video URL", placeholder="https://youtu.be/...", label_visibility="collapsed")
    process_btn = st.button("📥 Process Video", type="primary", use_container_width=True)
    
    st.markdown("━━━━━━━━━━━━━━━━━━━━━━")
    st.markdown("### ⚙️ System Config")
    st.caption("**Model**")
    st.markdown("`Llama 3.1 8B Instruct`")
    st.caption("**Embedding**")
    st.markdown("`NVIDIA Nemotron 1B`")

    if process_btn:
        if not url:
            st.error("Please enter a valid YouTube URL.")
        else:
            # Step-by-step progress UI
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.markdown("⏳ **Step 1/3:** Fetching Transcript...")
                progress_bar.progress(33)
                time.sleep(0.5) # UI polish
                
                status_text.markdown("✂️ **Step 2/3:** Chunking Text...")
                progress_bar.progress(66)
                time.sleep(0.5)
                
                status_text.markdown("🧠 **Step 3/3:** Generating Embeddings (FAISS)...")
                
                # Call cached function
                v_store, n_chunks = process_youtube_video(url)
                
                # Update Session State
                st.session_state.vector_store = v_store
                st.session_state.num_chunks = n_chunks
                st.session_state.video_url = url
                st.session_state.messages = [] 
                
                progress_bar.progress(100)
                status_text.success("✅ Ready! You can now ask questions.")
                time.sleep(1) # Let user see success before rerun
                st.rerun()
                
            except Exception as e:
                status_text.error(f"❌ Error: {e}")
                progress_bar.empty()

# ==========================================
# 4. Main UI: Hero Header
# ==========================================
st.markdown('<p class="hero-title">YouTube AI Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Chat with any YouTube video using <span class="hero-tech">NVIDIA Llama 3.1 • LangChain • FAISS</span></p>', unsafe_allow_html=True)

if st.session_state.vector_store is not None:
    # Top Dashboard Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Transcript", "✔ Loaded")
    col2.metric("🧩 Chunks", st.session_state.num_chunks)
    col3.metric("🧠 Model", "Llama 3.1")
    col4.metric("🔍 Retrieval", "Top 2")
    
    st.divider()

    # Video Preview & Chat Layout
    preview_col, chat_col = st.columns([1, 1.5], gap="large")
    
    with preview_col:
        st.markdown("### 📺 Video Preview")
        st.video(st.session_state.video_url)
        
        # Example Questions Buttons
        st.markdown("<br><b>💡 Quick Questions:</b>", unsafe_allow_html=True)
        q1, q2 = st.columns(2)
        if q1.button("🟦 Summarize this video", use_container_width=True):
            st.session_state.quick_question = "Can you provide a detailed summary of this video?"
        if q2.button("🟦 List key concepts", use_container_width=True):
            st.session_state.quick_question = "What are the main concepts or bullet points discussed?"
    
    with chat_col:
        st.markdown("### 💬 Chat")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Render sources if they exist in the message history
                if "sources" in message:
                    with st.expander("📚 Sources & Citations"):
                        for i, chunk in enumerate(message["sources"], 1):
                            st.markdown(f"**▶ Chunk {i}:**\n{chunk}")
                
        # Handle Chat Input (either manual text or quick button click)
        prompt_text = st.chat_input("E.g. What is RAG?")
        if st.session_state.quick_question:
            prompt_text = st.session_state.quick_question
            st.session_state.quick_question = None # Reset
            
        if prompt_text:
            st.chat_message("user").markdown(prompt_text)
            st.session_state.messages.append({"role": "user", "content": prompt_text})

            with st.chat_message("assistant"):
                with st.spinner("Analyzing context..."):
                    try:
                        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 2})
                        docs = retriever.invoke(prompt_text)
                        context = "\n\n".join(doc.page_content for doc in docs)
                        sources = [doc.page_content for doc in docs]

                        llm = ChatNVIDIA(
                            model="meta/llama-3.1-8b-instruct",
                            api_key=os.getenv("NVIDIA_API_KEY"),
                            timeout=120
                        )

                        prompt_template = ChatPromptTemplate.from_template("""
                        You are a helpful AI assistant.
                        Answer the user's question ONLY from the provided context.
                        If the answer is not present in the context, reply: "I don't know based on the provided context."

                        Context:
                        {context}

                        Question:
                        {question}
                        """)

                        formatted_prompt = prompt_template.invoke({
                            "context": context,
                            "question": prompt_text
                        })

                        response = llm.invoke(formatted_prompt)
                        st.markdown(response.content)
                        
                        # Source Citations
                        with st.expander("📚 Sources & Citations"):
                            for i, chunk in enumerate(sources, 1):
                                st.markdown(f"**▶ Chunk {i}:**\n{chunk[:400]}...")

                        # Save to history with sources
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response.content,
                            "sources": [chunk[:400] + "..." for chunk in sources]
                        })

                    except Exception as e:
                        st.error(f"❌ An error occurred: {e}")
else:
    # Empty State
    st.info("👈 Paste a YouTube URL in the sidebar and click **Process Video** to get started.")

# ==========================================
# 5. Footer
# ==========================================
st.markdown('<div class="footer">Powered by NVIDIA NIM • LangChain • FAISS • Streamlit</div>', unsafe_allow_html=True)