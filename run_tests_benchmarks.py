#!/usr/bin/env python
"""
Easy runner script for reshape and sum tests and benchmarks.

This script provides a convenient way to run tests and benchmarks
for the Tensor reshape and sum operations.

Usage:
    python run_tests_benchmarks.py [command]

Commands:
    test                    - Run reshape & sum tests only
    test-all                - Run all tensor tests
    benchmark               - Run full benchmarks (may take several minutes)
    benchmark-quick         - Run quick benchmarks with fewer iterations
    benchmark-no-pytorch    - Run benchmarks without PyTorch
    benchmark-no-numpy      - Run benchmarks without NumPy
    benchmark-litetorch-only- Run benchmarks for LiteTorch only
    all                     - Run all tests and benchmarks
    install                 - Install dependencies
    help                    - Show help message
"""

import sys
import os
import subprocess


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color
    
    @classmethod
    def disable(cls):
        """Disable colors (for Windows or non-TTY)."""
        cls.GREEN = ''
        cls.BLUE = ''
        cls.YELLOW = ''
        cls.RED = ''
        cls.NC = ''


# Disable colors on Windows or non-TTY
if os.name == 'nt' or not sys.stdout.isatty():
    Colors.disable()


def print_header(text):
    """Print a section header."""
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print(f"{Colors.BLUE}  {text}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.NC}\n")


def print_success(text):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")


def print_error(text):
    """Print an error message."""
    print(f"{Colors.RED}✗ {text}{Colors.NC}")


def print_info(text):
    """Print an info message."""
    print(f"{Colors.YELLOW}ℹ {text}{Colors.NC}")


def run_command(cmd, env=None):
    """Run a command and return the exit code."""
    if env:
        env_vars = os.environ.copy()
        env_vars.update(env)
    else:
        env_vars = None
    
    result = subprocess.run(cmd, shell=True, env=env_vars)
    return result.returncode


def cmd_test():
    """Run reshape & sum tests only."""
    print_header("Running Reshape & Sum Tests")
    print_info("Running comprehensive test suite for reshape and sum operations...")
    
    code = run_command("python -m pytest tests/test_reshape_sum.py -v")
    if code == 0:
        print_success("Tests completed!")
    else:
        print_error("Tests failed!")
    return code


def cmd_test_all():
    """Run all tensor tests."""
    print_header("Running All Tensor Tests")
    print_info("Running all tensor test suites...")
    
    code = run_command("python -m pytest tests/test_tensor.py tests/test_reshape_sum.py -v")
    if code == 0:
        print_success("All tests completed!")
    else:
        print_error("Some tests failed!")
    return code


def cmd_benchmark():
    """Run full benchmarks."""
    print_header("Running Reshape & Sum Benchmarks")
    print_info("Benchmarking reshape and sum operations against NumPy and PyTorch...")
    print_info("This may take a few minutes...")
    
    code = run_command("python benchmarks/bench_reshape_sum.py")
    if code == 0:
        print_success("Benchmarks completed!")
    else:
        print_error("Benchmarks failed!")
    return code


def cmd_benchmark_quick():
    """Run quick benchmarks with fewer iterations."""
    print_header("Running Quick Benchmarks")
    print_info("Running quick benchmarks with fewer iterations...")
    
    code = run_command("python benchmarks/bench_reshape_sum.py", env={"ITERATIONS": "100"})
    if code == 0:
        print_success("Quick benchmarks completed!")
    else:
        print_error("Quick benchmarks failed!")
    return code


def cmd_benchmark_no_pytorch():
    """Run benchmarks without PyTorch."""
    print_header("Running Benchmarks (NumPy only)")
    print_info("Running benchmarks without PyTorch...")
    
    code = run_command("python benchmarks/bench_reshape_sum.py", env={"SKIP_PYTORCH": "1"})
    if code == 0:
        print_success("Benchmarks completed!")
    else:
        print_error("Benchmarks failed!")
    return code


def cmd_benchmark_no_numpy():
    """Run benchmarks without NumPy."""
    print_header("Running Benchmarks (PyTorch only)")
    print_info("Running benchmarks without NumPy...")
    
    code = run_command("python benchmarks/bench_reshape_sum.py", env={"SKIP_NUMPY": "1"})
    if code == 0:
        print_success("Benchmarks completed!")
    else:
        print_error("Benchmarks failed!")
    return code


def cmd_benchmark_litetorch_only():
    """Run benchmarks for LiteTorch only."""
    print_header("Running Benchmarks (LiteTorch only)")
    print_info("Running benchmarks for LiteTorch only...")
    
    code = run_command("python benchmarks/bench_reshape_sum.py", 
                      env={"SKIP_PYTORCH": "1", "SKIP_NUMPY": "1"})
    if code == 0:
        print_success("Benchmarks completed!")
    else:
        print_error("Benchmarks failed!")
    return code


def cmd_all():
    """Run all tests and benchmarks."""
    print_header("Running All Tests and Benchmarks")
    
    print_info("Step 1/2: Running tests...")
    code = run_command("python -m pytest tests/test_reshape_sum.py -v")
    if code != 0:
        print_error("Tests failed!")
        return code
    print_success("Tests passed!")
    
    print_info("Step 2/2: Running benchmarks...")
    code = run_command("python benchmarks/bench_reshape_sum.py")
    if code != 0:
        print_error("Benchmarks failed!")
        return code
    print_success("Benchmarks completed!")
    
    print_success("All tasks completed successfully!")
    return 0


def cmd_install():
    """Install dependencies."""
    print_header("Installing Dependencies")
    print_info("Installing required packages...")
    
    code = run_command("pip install -e .")
    if code != 0:
        print_error("Failed to install required packages!")
        return code
    
    print_info("Installing optional benchmarking dependencies...")
    code = run_command("pip install torch --index-url https://download.pytorch.org/whl/cpu")
    if code != 0:
        print_error("Failed to install PyTorch (optional)")
    
    print_success("Installation completed!")
    return 0


def cmd_help():
    """Show help message."""
    print(__doc__)
    return 0


def main():
    """Main entry point."""
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Get command
    command = sys.argv[1] if len(sys.argv) > 1 else "help"
    
    # Command map
    commands = {
        "test": cmd_test,
        "test-all": cmd_test_all,
        "benchmark": cmd_benchmark,
        "benchmark-quick": cmd_benchmark_quick,
        "benchmark-no-pytorch": cmd_benchmark_no_pytorch,
        "benchmark-no-numpy": cmd_benchmark_no_numpy,
        "benchmark-litetorch-only": cmd_benchmark_litetorch_only,
        "all": cmd_all,
        "install": cmd_install,
        "help": cmd_help,
    }
    
    # Execute command
    if command in commands:
        exit_code = commands[command]()
        sys.exit(exit_code)
    else:
        print_error(f"Unknown command: {command}")
        cmd_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
