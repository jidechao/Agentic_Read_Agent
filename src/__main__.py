"""CLI 入口: python -m src compile"""
import sys

from src.triggers.cli import main

if __name__ == "__main__":
    sys.exit(main())
