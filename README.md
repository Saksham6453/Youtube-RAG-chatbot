# 🎥 YouTube AI Assistant (RAG)

> Chat with any YouTube video using **NVIDIA Llama 3.1**, **NVIDIA Embeddings**, **LangChain**, and **FAISS**.


---

## 📌 Overview

YouTube AI Assistant is a **Retrieval-Augmented Generation (RAG)** application that allows users to ask natural language questions about any YouTube video.

The application automatically:

- 🎥 Extracts the YouTube transcript
- ✂ Splits the transcript into semantic chunks
- 🧠 Generates embeddings using NVIDIA Embedding Model
- 📚 Stores embeddings inside FAISS Vector Database
- 🔍 Retrieves the most relevant chunks
- 🤖 Uses NVIDIA Llama 3.1 to generate context-aware answers

---

# 🚀 Demo

### 🌐 Live Demo

> https://YOUR-STREAMLIT-URL.streamlit.app


---

# ✨ Features

- 🎥 Chat with any YouTube Video
- 🧠 NVIDIA Llama 3.1 Instruct Model
- 📚 NVIDIA Llama Nemotron Embeddings
- ⚡ FAISS Vector Database
- 🔍 Semantic Search
- 📖 Context-aware Question Answering
- 📑 Source & Citation Display
- 🎨 Modern Streamlit UI
- 💬 Conversational Chat Interface
- 📺 YouTube Video Preview
- 📊 AI Pipeline Information Sidebar

---

# 🏗 Architecture

```

                User
                  │
                  ▼
         Enter YouTube URL
                  │
                  ▼
        YouTube Transcript Loader
                  │
                  ▼
      Recursive Character Splitter
                  │
                  ▼
      NVIDIA Embedding Model
                  │
                  ▼
            FAISS Vector DB
                  │
                  ▼
           Similarity Search
                  │
                  ▼
        Retrieved Context Chunks
                  │
                  ▼
         Prompt Template
                  │
                  ▼
      NVIDIA Llama 3.1 LLM
                  │
                  ▼
            Final Response

```

---

# ⚙ Tech Stack

| Technology | Usage |
|------------|-------|
| Python | Programming Language |
| Streamlit | Frontend |
| LangChain | RAG Framework |
| NVIDIA NIM | LLM + Embeddings |
| FAISS | Vector Database |
| YouTube Transcript API | Transcript Extraction |
| Python Dotenv | Environment Variables |

---

# 📂 Project Structure

```

YT-Rag-Project/

│

├── app.py

├── requirements.txt

├── .env.example

├── assets/

│   ├── ChatInterface.png

│   ├── Chunks.png


│

├── faiss_db/

│

└── README.md

```

---

# ⚡ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/youtube-ai-rag.git

cd youtube-ai-rag
```

Create virtual environment

```bash
python -m venv .venv
```

Activate virtual environment

Windows

```bash
.venv\Scripts\activate
```

Linux / Mac

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file

```env
NVIDIA_API_KEY=your_nvidia_api_key
```

---

# ▶ Run the Project

```bash
streamlit run app.py
```

---

# 💻 How It Works

### Step 1

Paste a YouTube URL.

↓

### Step 2

Transcript is automatically fetched.

↓

### Step 3

Transcript is divided into chunks.

↓

### Step 4

Chunks are converted into vector embeddings.

↓

### Step 5

Vectors are stored inside FAISS.

↓

### Step 6

Relevant chunks are retrieved.

↓

### Step 7

The retrieved context is passed to NVIDIA Llama.

↓

### Step 8

The LLM generates an accurate answer.

---

# 📊 RAG Pipeline

```

User Question

↓

Retriever

↓

Relevant Chunks

↓

Prompt Template

↓

LLM

↓

Final Answer

```

---

# 📈 Future Improvements

- ✅ Multi-Video Chat
- ✅ PDF + YouTube RAG
- ✅ Conversation Memory
- ✅ Streaming Responses
- ✅ Citation Highlighting
- ✅ Voice Input
- ✅ Voice Output
- ✅ Multi-language Support
- ✅ Persistent Vector Database
- ✅ User Authentication

---

# 👨‍💻 Author

**Saksham Sharma**

MCA (AI & ML)

AI Engineer


---
