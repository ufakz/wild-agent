# Wild Agent 🦁

**Sample Collection and Similarity Ranking using LLMs and Semantic Embeddings**

Wild Agent analyzes your text samples, collects similar content from the web using LLMs and web crawling, and ranks results by semantic similarity using state-of-the-art embeddings.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **🔍 Intelligent Theme Extraction**: Analyzes your samples using LLMs (Grok API) to extract common themes and patterns
- **🌐 Multi-Source Collection**: Gathers similar content from:
  - Online search powered by LLMs
  - URL crawling with async web scraping (crawl4ai)
- **🧹 Smart Deduplication**: Removes duplicates using:
  - Hash-based exact duplicate detection
  - Embedding-based near-duplicate detection (>95% similarity)
- **📊 Semantic Ranking**: Ranks candidates using Google's EmbeddingGemma (768-dim embeddings)
- **⚡ Async Processing**: Concurrent operations for fast collection and ranking
- **🎨 Beautiful CLI**: Rich console output with progress indicators and similarity visualizations
- **📦 JSON Export**: Export results for integration with other tools

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wild-agent.git
cd wild-agent

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys:
# - XAI_API_KEY: Your Grok API key (required for LLM features)
# - HF_TOKEN: Your HuggingFace token (required for EmbeddingGemma)
#   Get it from: https://huggingface.co/settings/tokens
#   Accept license at: https://huggingface.co/google/embeddinggemma-300m
```

### Basic Usage

```bash
# Analyze a single sample and find similar content
wild-agent collect --sample "Your reference text here"

# Multiple samples with specific URLs
wild-agent collect \
  --sample "First sample" \
  --sample "Second sample" \
  --url https://example.com \
  --url https://another-site.com

# Load from files and export results
wild-agent collect \
  --sample-file samples.txt \
  --urls-file urls.txt \
  --export results.json \
  --verbose

# Adjust ranking parameters
wild-agent collect \
  --sample "Reference text" \
  --threshold 0.8 \
  --top-n 5
```

## 📖 Documentation

### Command-Line Interface

```
wild-agent collect [OPTIONS]

Options:
  -s, --sample TEXT          Text sample to analyze (multiple allowed)
  -f, --sample-file PATH     File containing samples (one per line)
  -u, --url TEXT            URL to crawl (multiple allowed)
  --urls-file PATH          File containing URLs (one per line)
  -n, --top-n INTEGER       Maximum results to return [default: 10]
  -t, --threshold FLOAT     Minimum similarity threshold 0.0-1.0 [default: 0.7]
  --no-crawl                Disable URL crawling
  --no-online-search        Disable online search
  -e, --export PATH         Export results to JSON file
  -v, --verbose             Enable verbose logging
  --help                    Show this message and exit
```

### File Formats

**samples.txt** - One sample per line, `#` for comments:
```
# Reference samples
This is my first reference text about machine learning.
This is another sample discussing neural networks.
# Add more samples below
```

**urls.txt** - One URL per line, `#` for comments:
```
# URLs to crawl
https://example.com/article1
https://example.com/article2
# Add more URLs below
```

### Environment Variables

```bash
# Required
export XAI_API_KEY="your-grok-api-key"
export HF_TOKEN="your-huggingface-token"

# Optional (with defaults)
export XAI_API_BASE="https://api.x.ai/v1"
export XAI_MODEL="grok-beta"
export LLM_TEMPERATURE="0.7"
export LLM_MAX_TOKENS="2000"
export LLM_TIMEOUT="30.0"
export LLM_MAX_RETRIES="3"
```

**Note**: You need to accept the EmbeddingGemma license at [https://huggingface.co/google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) before using the model.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Interface                            │
│  (click, argument parsing, output formatting)                   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestrator                         │
└─────────┬─────────────────┬─────────────────┬───────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│  LLM Analyzer    │ │   Sample     │ │   Similarity     │
│  (Grok API)      │ │  Collector   │ │    Ranker        │
│                  │ │              │ │                  │
│ • Theme extract  │ │ • Online     │ │ • Embedding      │
│ • JSON parsing   │ │   search     │ │   generation     │
│ • Error handling │ │ • Web crawl  │ │ • Cosine sim     │
│                  │ │ • Dedup      │ │ • Threshold      │
└──────────────────┘ └──────────────┘ └──────────────────┘
```

### Core Components

1. **LLM Analyzer** (`src/analyzers/`)
   - Extracts themes from reference samples
   - Uses Grok API for analysis
   - Configurable via environment variables

2. **Sample Collector** (`src/collectors/`)
   - **Online Collector**: LLM-powered internet search
   - **Crawler Collector**: Async web crawling with crawl4ai
   - **Orchestrator**: Coordinates both sources in parallel

3. **Deduplicator** (`src/lib/`)
   - Stage 1: SHA-256 hash for exact duplicates
   - Stage 2: Embedding similarity for near-duplicates

4. **Similarity Ranker** (`src/rankers/`)
   - **Embedding Service**: Google EmbeddingGemma-300m (768-dim vectors)
   - **Similarity Calculator**: Cosine/Euclidean/Dot Product
   - **Ranker**: Filters, sorts, and limits results

5. **Data Models** (`src/models/`)
   - Pydantic models with validation
   - 12 models covering all entities

## 🔧 Development

### Setup Development Environment

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Format code
ruff format .
```

### Project Structure

```
wild-agent/
├── src/
│   ├── analyzers/          # LLM analysis components
│   ├── collectors/         # Sample collection (online + crawler)
│   ├── rankers/           # Similarity ranking
│   ├── models/            # Pydantic data models
│   ├── cli/               # Command-line interface
│   └── lib/               # Utilities (deduplication, etc.)
├── tests/
│   ├── unit/              # Unit tests for models
│   ├── contract/          # Contract tests for components
│   └── integration/       # End-to-end tests
├── specs/                 # Design documents
├── examples/              # Example files
└── pyproject.toml         # Project configuration
```

### Running Tests

```bash
# All tests
pytest

# Specific test category
pytest tests/unit/
pytest tests/contract/
pytest tests/integration/

# With coverage
pytest --cov=src --cov-report=html

# Verbose output
pytest -v
```

## 📊 Example Output

```
🔍 Analyzing Samples
  Extracting themes from 2 sample(s)...
  ✓ Analysis complete

🌐 Collecting Samples
  Using: online search, URL crawling (3 URLs)
  ✓ Collection complete
  Total samples: 15
  Duplicates removed: 3
  
📊 Ranking Candidates
  Ranking 12 candidates...
  Threshold: 0.70 | Top-N: 10
  ✓ Ranking complete
  Ranked results: 8

🏆 Top Results

#1 [████████████████████] 0.912
  Machine learning is a subset of artificial intelligence that enables...
  Source: INTERNET_SEARCH | https://example.com/ml-intro

#2 [██████████████████░░] 0.874
  Neural networks are computing systems inspired by biological networks...
  Source: URL_CRAWL | https://another-site.com/neural-nets

...

📈 Summary
  Reference samples: 2
  Candidates collected: 15
  Results ranked: 8
  Total time: 12.45s
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Write tests** for your changes
4. **Run linting**: `ruff check . && ruff format .`
5. **Commit changes**: `git commit -am 'Add feature'`
6. **Push to branch**: `git push origin feature/your-feature`
7. **Open a Pull Request**

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Write docstrings for public APIs
- Maintain test coverage above 80%

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google EmbeddingGemma**: For state-of-the-art semantic embeddings
- **sentence-transformers**: For the embedding framework
- **crawl4ai**: For async web crawling
- **Grok (X.AI)**: For LLM capabilities
- **Pydantic**: For data validation
- **click**: For CLI framework

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/wild-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/wild-agent/discussions)

---

**Made with ❤️ by the Wild Agent Team**
