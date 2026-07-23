#!/usr/bin/env python
"""FunASR API Server launcher script.

Usage:
    python run.py --model sensevoice --device cuda --port 8000
    python run.py --engine vllm --model fun-asr-nano --device cuda
    uv run funasr-api --model sensevoice --device cpu
"""

from app.main import main

if __name__ == "__main__":
    main()
