# Chat-Enabled Dataroom for Real Estate Documents

A prototype AI-powered dataroom that allows users to upload real estate documents and ask natural language questions about them. Built with LangChain Agent framework and Google Gemini for intelligent document analysis and Q&A.

## 🚀 Features

- **Document Upload**: Support for PDF, CSV, DOCX, and TXT files
- **Intelligent Q&A**: Natural language questions with AI-powered answers
- **Citation Tracking**: Every answer includes explicit citations (document name, page, section)
- **Agent-Based Architecture**: Multi-step reasoning for complex queries
- **Web Interface**: Clean Gradio-based UI for easy interaction

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gradio UI     │    │  LangChain      │    │   ChromaDB      │
│                 │◄──►│     Agent       │◄──►│  Vector Store   │
│  - File Upload  │    │                 │    │                 │
│  - Chat Interface│   │  - Tools        │    │  - Embeddings   │
└─────────────────┘    │  - Memory       │    │  - Metadata     │
                       │  - Reasoning    │    └─────────────────┘
                       └─────────────────┘
                                │
                       ┌─────────────────┐
                       │  Google Gemini  │
                       │     LLM         │
                       │                 │
                       └─────────────────┘
```

## 📁 Project Structure

```
dataroom/
├── src/
│   ├── dataroom/
│   │   ├── __init__.py
│   │   ├── main.py                 # Main application entry point
│   │   ├── config.py               # Configuration management
│   │   ├── agent/
│   │   │   ├── __init__.py
│   │   │   ├── real_estate_agent.py # Main agent implementation
│   │   │   ├── memory.py           # Agent memory management
│   │   │   └── reasoning.py        # Reasoning logic
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── document_search.py  # Document search tool
│   │   │   ├── calculation.py      # Calculation tool
│   │   │   ├── citation.py         # Citation generation tool
│   │   │   └── document_processor.py # Document processing tool
│   │   ├── prompts/
│   │   │   ├── __init__.py
│   │   │   ├── system_prompts.py   # System prompts for agent
│   │   │   ├── tool_prompts.py     # Tool-specific prompts
│   │   │   └── citation_prompts.py # Citation generation prompts
│   │   ├── ui/
│   │   │   ├── __init__.py
│   │   │   ├── gradio_app.py       # Gradio interface
│   │   │   └── components.py       # UI components
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── file_utils.py       # File handling utilities
│   │       ├── logging.py          # Logging configuration
│   │       └── validators.py       # Input validation
├── data/
│   ├── uploads/                    # Uploaded documents
│   ├── processed/                  # Processed documents
│   └── chroma_db/                  # ChromaDB storage
├── tests/
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_tools.py
│   └── test_ui.py
├── prompts/
│   ├── system_prompt.txt
│   ├── tool_descriptions.txt
│   └── citation_template.txt
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── deployment.md
├── .gitignore
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd dataroom
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your Google API key
   ```

5. **Run the application**
   ```bash
   python src/dataroom/main.py
   ```

## 🔧 Configuration

Key configuration options in `.env`:

- `GOOGLE_API_KEY`: Your Google Gemini API key
- `CHROMA_PERSIST_DIRECTORY`: Path for ChromaDB storage
- `MAX_FILE_SIZE_MB`: Maximum file size for uploads
- `CHUNK_SIZE`: Document chunking size for processing

## 🎯 Usage

1. **Upload Documents**: Use the web interface to upload PDF, CSV, or other supported documents
2. **Ask Questions**: Type natural language questions about your documents
3. **Get Answers**: Receive AI-powered answers with explicit citations

### Example Questions:
- "What is the total rental income from all leases?"
- "Which tenant has the highest rent per square foot?"
- "What are the key terms in the lease agreement for tenant ABC?"
- "Calculate the average lease duration across all contracts"

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/dataroom

# Run specific test file
pytest tests/test_agent.py
```

## 📝 Development

### Code Quality
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
```

## 🚀 Deployment

The application can be deployed using:
- Docker
- Cloud platforms (Google Cloud, AWS, Azure)
- Local server deployment

See `docs/deployment.md` for detailed deployment instructions.

## 📊 Performance Considerations

- **Document Processing**: Optimized chunking strategies for large documents
- **Vector Search**: Efficient similarity search with ChromaDB
- **Memory Management**: Agent memory optimization for long conversations
- **Caching**: Response caching for frequently asked questions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔮 Future Enhancements

- [ ] Multi-modal document support (images, tables)
- [ ] Advanced analytics and reporting
- [ ] Multi-language support
- [ ] API endpoints for integration
- [ ] Advanced security and access control
- [ ] Real-time collaboration features

## 📞 Support

For questions and support, please open an issue on GitHub or contact the development team.

---

**Note**: This is a prototype project for evaluation purposes. Focus is on demonstrating approach, solution structure, and critical design decisions rather than feature completeness.
