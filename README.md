# JEE Bench Q&A System

A Question Answering system for JEE (Joint Entrance Examination) problems, built using:

- **[Tavily](https://tavily.com/)**: For web search to find relevant educational content
- **[Qdrant](https://qdrant.tech/)**: Vector database for semantic search
- **[LangChain](https://www.langchain.com/)**: Framework for building LLM-powered applications
- **[LangGraph](https://github.com/langchain-ai/langgraph)**: DAG orchestration for LLM workflows
- **[Groq](https://groq.com/)**: LLM provider for fast responses
- **[HuggingFace](https://huggingface.co/)**: For text embeddings

The system uses the JEE Bench dataset to demonstrate advanced question answering capabilities for Physics, Chemistry, and Mathematics problems.

## Features

- Semantic search for finding similar questions in the JEE Bench dataset
- Web search for retrieving relevant educational content
- Few-shot prompting with subject and question type matching
- Multiple Qdrant deployment options (in-memory, local, cloud)
- Text chunking for improved semantic search
- Persistence options for in-memory Qdrant
- Automatic vector store reuse between application runs
- Streamlit web interface for interactive exploration
- Command-line interface for programmatic usage
- Comprehensive workflow orchestration using LangGraph

## Project Structure

```
├── app/                    # Web application
│   └── streamlit_app.py    # Streamlit application
├── config/                 # Configuration
│   └── config.py           # Configuration settings
├── data/                   # Data files
│   ├── dataset.json        # JEE Bench dataset
│   ├── few_shot_examples.json  # Few-shot examples
│   └── responses/          # Example responses
├── scripts/                # Utility scripts
│   └── embedding_example.py # HuggingFace embeddings example
├── src/                    # Source code
│   ├── data_loader.py      # Data loading utilities
│   ├── groq_llm.py         # Groq LLM integration
│   ├── setup.py            # Setup utilities
│   ├── tavily_search.py    # Tavily search integration
│   ├── vector_store.py     # Qdrant integration
│   └── workflow.py         # LangGraph workflow
├── app.py                  # Command-line application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```
## overview
System Architecture Overview
```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│                 │     │                  │     │                      │
│  User Interface │────▶│   AI Gateway    │────▶│   Routing Agent      │
│  (Streamlit)    │     │   (Guardrails)   │     │                      │
│                 │◀────│                  │◀────│                      │
└─────────────────┘     └──────────────────┘     └──────────────────────┘
                                                           │
                                                           ▼
                       ┌──────────────────────────────────────────────────┐
                       │                                                  │
                       │                Decision Router                   │
                       │                                                  │
                       └──────────────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┴───────────────────────┐
                    │                                              │
                    ▼                                              ▼
    ┌─────────────────────────┐                       ┌──────────────────────┐
    │                         │                       │                      │
    │  Knowledge Base Agent   │                       │   Web Search Agent   │
    │  (ChromaDB + JEE Bench) │                       │   (DuckDuckGo)       │
    │                         │                       │                      │
    └─────────────────────────┘                       └──────────────────────┘
                    │                                              │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                       ┌──────────────────────────────────────────────────┐
                       │                                                  │
                       │               Response Synthesizer               │
                       │               (Step-by-Step Solution)            │
                       │                                                  │
                       └──────────────────────────────────────────────────┘
                                           │
                                           ▼
                       ┌──────────────────────────────────────────────────┐
                       │                                                  │
                       │              Human Feedback Agent                │
                       │                         │
                       │                                                  │
                       └──────────────────────────────────────────────────┘
```
### 1. AI Gateway (Guardrails)
The AI Gateway provides input and output guardrails:

Input Guardrails:

Restrict to mathematics-related queries only
Filter out potentially harmful, offensive, or inappropriate content
Ensure questions are well-formed and contain sufficient context


Output Guardrails:

Ensure responses are educationally appropriate
Verify that solutions follow a step-by-step approach
Remove any content unrelated to mathematics



### 2. Knowledge Base Creation

Vector Database: ChromaDB
Dataset: JEE Bench dataset (contains IIT-JEE mathematics problems and solutions)
Embeddings: HuggingFace embeddings model (e.g., SBERT/MPNet)
Storage Structure:

Questions and solutions indexed for efficient retrieval
Metadata including topics, difficulty levels, and solution approaches



### 3. Web Search Capability

Search Engine: DuckDuckGo API
Search Strategy:

Query reformation for maximum relevance
Result filtering for mathematical content
Credibility assessment of sources
Content extraction and summarization



### 4. Human-in-the-Loop Feedback Mechanism

Feedback Collection:

User ratings on solution correctness (1-5 scale)
Specific feedback on steps that need improvement
Alternative solution approaches suggested by users


Learning Pipeline:

Store feedback in a feedback database
Use feedback to refine the solution generation process
Update retrieval strategy based on which sources led to correct solutions



### 5. LLM Integration

Primary LLM: Groq API
Framework: LangChain Community
Prompt Engineering:

Structured prompts for each system component
Chain-of-thought reasoning for step-by-step solutions



### Implementation Approach

Routing Logic:

First attempt to answer from knowledge base
If confidence score < threshold, proceed to web search
If web search yields insufficient information, indicate limitations


Solution Generation Strategy:

Parse problem statement to identify mathematical domain
Break down complex problems into smaller sub-problems
Generate progressive solution steps with explanations
Include visualizations where applicable


Feedback Integration:

Use feedback to fine-tune prompts
Adjust retrieval relevance scores based on user feedback
Create a growing repository of verified solutions
## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables (API keys):
   - Create a `.env` file in the project root with:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   HUGGINGFACE_API_KEY=your_huggingface_api_key_here  # Only needed if using huggingface-api embedding model
   
   # Optional configurations
   EMBEDDING_MODEL=huggingface-local  # Options: huggingface-local (default), huggingface-api
   GROQ_MODEL=llama3-8b-8192  # Default LLM model to use
   
   # Vector DB Configuration
   QDRANT_MODE=in-memory  # Options: in-memory (default), local, cloud
   QDRANT_HOST=localhost  # For local mode
   QDRANT_PORT=6333  # For local mode
   QDRANT_URL=  # For cloud mode
   QDRANT_API_KEY=  # For cloud mode
   QDRANT_COLLECTION=jee_bench_data
   QDRANT_PERSIST_DIR=qdrant_data  # Directory for persisting in-memory data
   
   # Chunking Configuration
   USE_CHUNKING=False  # Set to True to enable text chunking
   CHUNK_SIZE=1000  # Size of each text chunk
   CHUNK_OVERLAP=200  # Overlap between chunks
   ```

4. Choose your Qdrant deployment:

   **Option 1: In-memory** (default, no installation required):
   ```
   QDRANT_MODE=in-memory
   ```
   Optionally set `QDRANT_PERSIST_DIR` to persist data to disk.
   
   **Option 2: Local Qdrant server** (requires installation):
   ```
   QDRANT_MODE=local
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   ```
   
   **Using Docker** (recommended for local mode):
   ```bash
   docker pull qdrant/qdrant
   docker run -p 6333:6333 -p 6334:6334 -v /path/to/qdrant_storage:/qdrant/storage qdrant/qdrant
   ```
   
   **Without Docker**:
   - Windows: Download the latest release from [Qdrant releases](https://github.com/qdrant/qdrant/releases)
   - Linux/macOS: Follow the [installation instructions](https://qdrant.tech/documentation/install/)
   
   **Option 3: Cloud** (Qdrant Cloud):
   ```
   QDRANT_MODE=cloud
   QDRANT_URL=https://your-deployment-url.qdrant.tech
   QDRANT_API_KEY=your_qdrant_api_key
   ```

5. Text Chunking (Optional):
   - Enable chunking to break down large questions into smaller pieces:
   ```
   USE_CHUNKING=True
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   ```

6. Run the setup process:
   ```bash
   python app.py --setup
   ```

## Usage

### Command-line Interface

Set up the application (only needed once with persistence enabled):
```bash
python app.py --setup
```

Force rebuilding the vector index even if it exists:
```bash
python app.py --setup --force-reindex
```

Answer a specific question:
```bash
python app.py --question "What is Planck's constant?"
```

Answer a question from the dataset by index:
```bash
python app.py --index 15
```

Save the answer to a file:
```bash
python app.py --index 15 --output answer.json
```

### HuggingFace Embeddings Example

To test the HuggingFace embeddings:
```bash
python scripts/embedding_example.py
```

This script demonstrates:
- How to use the embeddings from HuggingFace
- Comparison of semantic similarity between texts
- No API key required for local HuggingFace embeddings

### Web Interface

Run the Streamlit application:
```bash
cd app
streamlit run streamlit_app.py
```

Then follow the setup steps in the UI:
1. Verify API keys
2. Load the dataset
3. Set up the vector database
4. Start asking questions

## Example Workflow

1. User enters a JEE problem
2. System classifies the subject and question type
3. Web search retrieves relevant educational content
4. Vector search finds similar questions in the dataset
5. Few-shot example is selected based on subject and question type
6. Groq LLM generates a detailed solution with steps
7. Answer is presented to the user

## Configuration Options

### Vector Database Options

- **In-memory mode**: Fastest option, data is persisted between application runs when QDRANT_PERSIST_DIR is set
- **Local mode**: Requires running a Qdrant server locally
- **Cloud mode**: Uses Qdrant Cloud for storage and retrieval

The application automatically detects if a vector store already exists with data and reuses it, 
avoiding the need to rebuild the index every time. Use the `--force-reindex` option if you 
need to rebuild the index (e.g., after dataset changes).

### Text Chunking

Chunking breaks down large texts into smaller pieces for more precise semantic search:

- **USE_CHUNKING**: Enable/disable chunking (default: False)
- **CHUNK_SIZE**: Number of characters in each chunk (default: 1000)
- **CHUNK_OVERLAP**: Number of overlapping characters between chunks (default: 200)

## Requirements

- Python 3.9+
- Groq API key
- Tavily API key
- HuggingFace API key (optional, only if using the API version of embeddings)
- Qdrant (optional, can use in-memory mode without installation)

## License

This project is for educational purposes only. The JEE Bench dataset is used for demonstration.
