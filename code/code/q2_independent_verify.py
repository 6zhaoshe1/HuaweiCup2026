#!/usr/bin/env python3
# AI assistance: OpenAI Codex, OpenAI, 2026-09-24; team review required.
"""Compatibility entrypoint for revised independent numerical verification.

The former arbitrary OLS-vs-Huber tolerance check has been removed.
Implementation lives in q2_revision_verify.py and does not import modeling metrics.
"""
from q2_revision_verify import main

if __name__ == '__main__':
    main()
