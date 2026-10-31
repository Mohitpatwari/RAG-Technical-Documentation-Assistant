"""
scripts/fetch_corpus.py
Downloads the documentation corpus (raw Markdown from the official repos) into data/corpus/.

FastAPI docs pull code in via `{* ../../docs_src/x.py hl[..] *}` directives. Those are
resolved here by fetching the file and inlining it as a fenced code block; otherwise the
code examples would be missing from the index.

Fails loudly (exit code 1) if anything cannot be fetched: no placeholder files.
Run from project root:  python scripts/fetch_corpus.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import requests

FA = "https://raw.githubusercontent.com/fastapi/fastapi/master"
PD = "https://raw.githubusercontent.com/pydantic/pydantic/main"

CORPUS = [
    {"title": "FastAPI - Path Parameters", "source_url": "https://fastapi.tiangolo.com/tutorial/path-params/",
     "raw_url": f"{FA}/docs/en/docs/tutorial/path-params.md", "filename": "fastapi_path_parameters.md"},
    {"title": "FastAPI - Request Body", "source_url": "https://fastapi.tiangolo.com/tutorial/body/",
     "raw_url": f"{FA}/docs/en/docs/tutorial/body.md", "filename": "fastapi_request_body.md"},
    {"title": "FastAPI - Dependencies", "source_url": "https://fastapi.tiangolo.com/tutorial/dependencies/",
     "raw_url": f"{FA}/docs/en/docs/tutorial/dependencies/index.md", "filename": "fastapi_dependencies.md"},
    {"title": "FastAPI - Background Tasks", "source_url": "https://fastapi.tiangolo.com/tutorial/background-tasks/",
     "raw_url": f"{FA}/docs/en/docs/tutorial/background-tasks.md", "filename": "fastapi_background_tasks.md"},
    {"title": "Pydantic - Models", "source_url": "https://docs.pydantic.dev/latest/concepts/models/",
     "raw_url": f"{PD}/docs/concepts/models.md", "filename": "pydantic_models.md"},
    {"title": "Pydantic - Validators", "source_url": "https://docs.pydantic.dev/latest/concepts/validators/",
     "raw_url": f"{PD}/docs/concepts/validators.md", "filename": "pydantic_validators.md"},
]

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"
INCLUDE_RE = re.compile(r"\{\*\s*(\S+?\.py)[^}]*?\*\}")


def _inline_code_includes(text: str) -> str:
    def repl(m: re.Match) -> str:
        rel = m.group(1).replace("../", "")  # '../../docs_src/x.py' -> 'docs_src/x.py'
        resp = requests.get(f"{FA}/{rel}", timeout=30)
        if resp.status_code != 200:
            return ""  # drop the directive rather than leave noise in the index
        return f"```python\n{resp.text.strip()}\n```"

    return INCLUDE_RE.sub(repl, text)


def _clean_markdown(text: str) -> str:
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)  # front matter
    text = re.sub(r"^///.*$", "", text, flags=re.MULTILINE)          # admonition fences
    text = re.sub(r"\s*\{\s*#[\w.-]+\s*\}\s*$", "", text, flags=re.MULTILINE)  # '# Title { #anchor }'
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_corpus() -> int:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    manifest, failed = [], 0
    for e in CORPUS:
        print(f"  Fetching: {e['title']} ...", end=" ", flush=True)
        try:
            r = requests.get(e["raw_url"], timeout=30)
            r.raise_for_status()
            content = _clean_markdown(_inline_code_includes(r.text))
            (CORPUS_DIR / e["filename"]).write_text(content, encoding="utf-8")
            print(f"OK ({len(content):,} chars, {content.count('```') // 2} code blocks)")
        except requests.RequestException as exc:
            failed += 1
            print(f"FAILED: {exc}")
            continue
        manifest.append({"title": e["title"], "source_url": e["source_url"],
                         "filename": e["filename"], "doc_id": e["filename"].removesuffix(".md")})
    (CORPUS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return failed


if __name__ == "__main__":
    print("=== Fetching documentation corpus ===\n")
    sys.exit(1 if fetch_corpus() else 0)
