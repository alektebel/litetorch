#!/bin/bash
# Easy runner script for tests and benchmarks

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print section headers
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Change to script directory
cd "$(dirname "$0")"

# Parse command line arguments
COMMAND=${1:-help}

case $COMMAND in
    test)
        print_header "Running Reshape & Sum Tests"
        print_info "Running comprehensive test suite for reshape and sum operations..."
        python -m pytest tests/test_reshape_sum.py -v
        print_success "Tests completed!"
        ;;
    
    test-all)
        print_header "Running All Tensor Tests"
        print_info "Running all tensor test suites..."
        python -m pytest tests/test_tensor.py tests/test_reshape_sum.py -v
        print_success "All tests completed!"
        ;;
    
    benchmark)
        print_header "Running Reshape & Sum Benchmarks"
        print_info "Benchmarking reshape and sum operations against NumPy and PyTorch..."
        print_info "This may take a few minutes..."
        python benchmarks/bench_reshape_sum.py
        print_success "Benchmarks completed!"
        ;;
    
    benchmark-quick)
        print_header "Running Quick Benchmarks"
        print_info "Running quick benchmarks with fewer iterations..."
        ITERATIONS=100 python benchmarks/bench_reshape_sum.py
        print_success "Quick benchmarks completed!"
        ;;
    
    benchmark-no-pytorch)
        print_header "Running Benchmarks (NumPy only)"
        print_info "Running benchmarks without PyTorch..."
        SKIP_PYTORCH=1 python benchmarks/bench_reshape_sum.py
        print_success "Benchmarks completed!"
        ;;
    
    benchmark-no-numpy)
        print_header "Running Benchmarks (PyTorch only)"
        print_info "Running benchmarks without NumPy..."
        SKIP_NUMPY=1 python benchmarks/bench_reshape_sum.py
        print_success "Benchmarks completed!"
        ;;
    
    benchmark-litetorch-only)
        print_header "Running Benchmarks (LiteTorch only)"
        print_info "Running benchmarks for LiteTorch only..."
        SKIP_PYTORCH=1 SKIP_NUMPY=1 python benchmarks/bench_reshape_sum.py
        print_success "Benchmarks completed!"
        ;;
    
    all)
        print_header "Running All Tests and Benchmarks"
        
        print_info "Step 1/2: Running tests..."
        python -m pytest tests/test_reshape_sum.py -v
        print_success "Tests passed!"
        
        print_info "Step 2/2: Running benchmarks..."
        python benchmarks/bench_reshape_sum.py
        print_success "Benchmarks completed!"
        
        print_success "All tasks completed successfully!"
        ;;
    
    install)
        print_header "Installing Dependencies"
        print_info "Installing required packages..."
        pip install -e .
        print_info "Installing optional benchmarking dependencies..."
        pip install torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || print_error "Failed to install PyTorch (optional)"
        print_success "Installation completed!"
        ;;
    
    help|*)
        echo "Usage: ./run_tests_benchmarks.sh [command]"
        echo ""
        echo "Commands:"
        echo "  test                    - Run reshape & sum tests only"
        echo "  test-all                - Run all tensor tests"
        echo "  benchmark               - Run full benchmarks (may take several minutes)"
        echo "  benchmark-quick         - Run quick benchmarks with fewer iterations"
        echo "  benchmark-no-pytorch    - Run benchmarks without PyTorch"
        echo "  benchmark-no-numpy      - Run benchmarks without NumPy"
        echo "  benchmark-litetorch-only- Run benchmarks for LiteTorch only"
        echo "  all                     - Run all tests and benchmarks"
        echo "  install                 - Install dependencies"
        echo "  help                    - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_tests_benchmarks.sh test"
        echo "  ./run_tests_benchmarks.sh benchmark-quick"
        echo "  ./run_tests_benchmarks.sh all"
        echo ""
        echo "Environment variables:"
        echo "  ITERATIONS=N            - Set number of iterations for benchmarks (default: 1000)"
        echo "  SKIP_PYTORCH=1          - Skip PyTorch benchmarks"
        echo "  SKIP_NUMPY=1            - Skip NumPy benchmarks"
        ;;
esac
