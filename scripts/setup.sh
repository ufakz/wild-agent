#!/bin/bash
#
# Wild Agent Setup Script
# Automated setup for dependencies and health checks
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Banner
echo ""
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}     Wild Agent Setup Script         ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Check Python version
print_step "Checking Python version..."
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD=python3.11
    print_success "Python 3.11 found"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
    MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)
    if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 11 ]; then
        PYTHON_CMD=python3
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.11+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3.11+ not found"
    echo "Please install Python 3.11 or higher:"
    echo "  macOS: brew install python@3.11"
    echo "  Ubuntu: sudo apt install python3.11"
    exit 1
fi

# Check if virtual environment exists
print_step "Checking virtual environment..."
if [ -d ".venv" ]; then
    print_warning "Virtual environment already exists"
    read -p "Recreate virtual environment? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        print_step "Creating virtual environment..."
        $PYTHON_CMD -m venv .venv
        print_success "Virtual environment created"
    fi
else
    print_step "Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_step "Activating virtual environment..."
source .venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_step "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

# Install dependencies
print_step "Installing project dependencies..."
pip install -e . > /dev/null 2>&1
print_success "Project dependencies installed"

# Install dev dependencies
read -p "Install development dependencies? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "Installing development dependencies..."
    pip install -e ".[dev]" > /dev/null 2>&1
    print_success "Development dependencies installed"
fi

# Check for API key
print_step "Checking API key configuration..."
if [ -z "$XAI_API_KEY" ]; then
    print_warning "XAI_API_KEY not set"
    echo ""
    echo "To use LLM features, set your Grok API key:"
    echo "  export XAI_API_KEY='your-api-key-here'"
    echo ""
    echo "Add to your shell profile (~/.bashrc, ~/.zshrc) to persist:"
    echo "  echo 'export XAI_API_KEY=\"your-api-key-here\"' >> ~/.zshrc"
    echo ""
else
    print_success "XAI_API_KEY is configured"
fi

# Download sentence-transformers model
print_step "Downloading sentence-transformers model..."
python3 -c "
from sentence_transformers import SentenceTransformer
import sys
try:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print('Model downloaded successfully')
except Exception as e:
    print(f'Error downloading model: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1 | grep -v "Some weights" | grep -v "huggingface"

if [ $? -eq 0 ]; then
    print_success "Sentence-transformers model ready"
else
    print_error "Failed to download model"
    exit 1
fi

# Create example files if they don't exist
print_step "Setting up example files..."
mkdir -p examples

if [ ! -f "examples/samples.txt" ]; then
    cat > examples/samples.txt << 'EOF'
# Example reference samples
# Add your samples below (one per line)

Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data.
Neural networks are computing systems inspired by biological neural networks that constitute animal brains.
Deep learning uses multiple layers to progressively extract higher-level features from raw input.
EOF
    print_success "Created examples/samples.txt"
else
    print_warning "examples/samples.txt already exists"
fi

if [ ! -f "examples/urls.txt" ]; then
    cat > examples/urls.txt << 'EOF'
# Example URLs to crawl
# Add your URLs below (one per line)

https://en.wikipedia.org/wiki/Machine_learning
https://en.wikipedia.org/wiki/Artificial_neural_network
https://en.wikipedia.org/wiki/Deep_learning
EOF
    print_success "Created examples/urls.txt"
else
    print_warning "examples/urls.txt already exists"
fi

# Run health checks
print_step "Running health checks..."
echo ""

# Check imports
python3 -c "
import sys
try:
    import click
    import httpx
    import pydantic
    import sentence_transformers
    import structlog
    import numpy
    import scipy
    print('✓ All required packages imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}', file=sys.stderr)
    sys.exit(1)
"

# Check CLI command
print_step "Testing CLI command..."
if command -v wild-agent &> /dev/null; then
    wild-agent --help > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        print_success "CLI command 'wild-agent' is working"
    else
        print_error "CLI command failed"
    fi
else
    print_warning "CLI command not in PATH"
    echo "  Run: pip install -e ."
fi

# Final summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}      Setup Complete! 🎉              ${GREEN}║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source .venv/bin/activate"
if [ -z "$XAI_API_KEY" ]; then
    echo "  2. Set API key: export XAI_API_KEY='your-key'"
fi
echo "  3. Try the CLI: wild-agent collect --sample 'Your text here'"
echo "  4. View examples: cat examples/samples.txt"
echo ""
echo "Documentation: README.md"
echo "Quickstart: specs/001-sample-collection-and/quickstart.md"
echo ""
