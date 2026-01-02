# Agentic RAG with CrewAI

An intelligent Retrieval-Augmented Generation (RAG) system powered by CrewAI that enables conversational search through your PDF documents with intelligent web search fallback.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![CrewAI](https://img.shields.io/badge/CrewAI-Latest-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

##  Features

- **PDF Document Search**: Upload and chat with your PDF documents using semantic search
- **Intelligent Web Fallback**: Automatically falls back to web search when PDF doesn't contain the answer
- **Local LLM Support**: Run entirely locally with Ollama (Llama 3.2, DeepSeek-R1)
- **Semantic Chunking**: Uses Chonkie's semantic chunking for better context preservation
- **Vector Store**: Qdrant in-memory vector store for fast similarity search
- **Agent-Based Architecture**: Multi-agent system with specialized retriever and synthesizer agents
- **Real-time Chat Interface**: Beautiful Streamlit UI for seamless interaction

##  Architecture

![Agentic RAG Architecture](assets/agentic-rag-architecture.png)


##  Prerequisites

- **Python 3.11+**
- **Ollama** installed and running (for local LLMs)
- **FireCrawl API Key** (for web search functionality)

##  Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd agentic_rag
```

### 2. Create Virtual Environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install crewai crewai-tools chonkie markitdown qdrant-client firecrawl-py streamlit
```

Or using the project configuration:

```bash
pip install -e .
```

### 4. Install Ollama

Download and install [Ollama](https://ollama.com/download)

Then pull the model you want to use:

```bash
# For Llama 3.2
ollama pull llama3.2

# For DeepSeek R1
ollama pull deepseek-r1
```

### 5. Get FireCrawl API Key

Sign up at [FireCrawl](https://www.firecrawl.dev/i/api) and get your API key.

##  Configuration

### Set Environment Variables

Create a `.env` file in the project root or set environment variables:

```bash
# FireCrawl API Key (required for web search)
FIRECRAWL_API_KEY=your-firecrawl-api-key-here

# Optional: Ollama base URL (default: http://localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434
```

### Choose Your LLM

The project supports two local LLM options:

1. **Llama 3.2** (Recommended for general use)
2. **DeepSeek-R1** (For reasoning tasks)

##  Usage

### Option 1: Using Llama 3.2

```bash
streamlit run app_llama3.2.py
```

### Option 2: Using DeepSeek R1

```bash
streamlit run app_deep_seek.py
```

The application will start at `http://localhost:8501`

### Using the Interface

1. **Upload a PDF**: Use the sidebar to upload your PDF document
2. **Indexing**: The system will automatically index your PDF (this may take a moment)
3. **Chat**: Ask questions about your PDF in the chat interface
4. **Web Fallback**: If the answer isn't in your PDF, the system will automatically search the web

##  Project Structure

```
agentic_rag/
├── src/
│   └── agentic_rag/
│       ├── __init__.py
│       ├── crew.py                 # CrewAI crew definition
│       ├── config/
│       │   ├── agents.yaml        # Agent configurations
│       │   └── tasks.yaml         # Task configurations
│       ├── tools/
│       │   ├── __init__.py
│       │   └── custom_tool.py    # Custom PDF & Web search tools
│       └── main.py               # Main entry point
├── app_llama3.2.py              # Streamlit app for Llama 3.2
├── app_deep_seek.py             # Streamlit app for DeepSeek R1
├── assets/                     # Static assets (logos, etc.)
├── knowledge/                  # Sample knowledge base (optional)
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

##  How It Works

### 1. Document Processing Pipeline

1. **PDF Upload**: User uploads a PDF through the Streamlit interface
2. **Text Extraction**: MarkItDown extracts text content from the PDF
3. **Semantic Chunking**: Chonkie's SemanticChunker breaks text into semantically meaningful chunks
4. **Embedding Generation**: Chonkie embeddings create vector representations
5. **Vector Storage**: Chunks are stored in Qdrant with their embeddings

### 2. Retrieval Process

1. **Query Embedding**: User query is converted to vector embeddings
2. **Similarity Search**: Qdrant finds the most similar chunks using cosine similarity
3. **Context Assembly**: Top-k chunks are assembled as context for the LLM

### 3. Agent Workflow

1. **Retriever Agent**: Decides whether to search PDF or web based on query
2. **Document Search**: Performs semantic search in the PDF if available
3. **Web Search**: Falls back to FireCrawl web search if PDF lacks information
4. **Synthesizer Agent**: Combines retrieved information into a coherent response


##  Technologies Used

- **[CrewAI](https://www.crewai.com/)**: Multi-agent orchestration framework
- **[Streamlit](https://streamlit.io/)**: Web application framework
- **[Ollama](https://ollama.com/)**: Local LLM runtime
- **[Chonkie](https://github.com/chonkie-ai/chonkie)**: Semantic text chunking
- **[Qdrant](https://qdrant.tech/)**: Vector similarity search engine
- **[MarkItDown](https://github.com/microsoft/markitdown)**: Document parsing
- **[FireCrawl](https://www.firecrawl.dev/)**: Web scraping and search

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

