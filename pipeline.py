import os
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

#INDEXING PHASE
# 1. Setup
load_dotenv()
url = input("Enter YouTube URL: ")

# 2. Load YouTube Transcript Automatically
print("⏳ Fetching transcript...")
# YoutubeLoader handles the URL parsing and transcript fetching for us!
loader = YoutubeLoader.from_youtube_url(
    url, add_video_info=False,
    language=["hi", "en"])
docs = loader.load()
print("✅ Transcript Loaded Successfully")

# 3. Split Text
splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f"✅ Total Chunks Created : {len(chunks)}")

# 4. Initialize Embeddings (It automatically grabs the API key from your .env)
embeddings = NVIDIAEmbeddings(
    model="nvidia/llama-nemotron-embed-1b-v2"
    )
print("✅ Embedding Model Initialized")

# 5. Create Vector Store 
print("⏳ Generating embeddings and building FAISS database...")
vector_store = FAISS.from_documents(
    chunks, 
    embeddings) 
vector_store.save_local("faiss_db")
print("✅ Successfully Stored all Chunks in FAISS")


print("\n🔍 Retrieval Phase Started...")
#RETRIEVAL PHASE
retriever = vector_store.as_retriever(
    search_kwargs = {"k":2}
)
#Query
question = input("Ask your question: ")
docs = retriever.invoke(question)
#Context
context = "\n\n".join(
    doc.page_content for doc in docs
)

#LLM / Augmentation phase
llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    api_key=os.getenv("NVIDIA_API_KEY"),
    timeout=120  # ⏳ Give the API up to 2 minutes to respond
)

#PROMPT
prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer the user's question ONLY from the provided context.

If the answer is not present in the context, reply:

"I don't know based on the provided context."

Context:
{context}

Question:
{question}
""")

formatted_prompt = prompt.invoke({
    "context": context,
    "question": question
})

# ====================================================
# GENERATE RESPONSE
# ====================================================

response = llm.invoke(formatted_prompt)

print("\n================ FINAL ANSWER ================\n")

print(response.content)
