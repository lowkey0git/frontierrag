"""Download a small, inspectable public-domain starter corpus."""
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
SOURCES = [
    ("frankenstein.txt", "https://www.gutenberg.org/cache/epub/84/pg84.txt", "Frankenstein; Mary Wollstonecraft Shelley; Project Gutenberg"),
    ("sherlock.txt", "https://www.gutenberg.org/cache/epub/1661/pg1661.txt", "The Adventures of Sherlock Holmes; Arthur Conan Doyle; Project Gutenberg"),
]
manifest = []
for filename, url, attribution in SOURCES:
    target = RAW / filename
    response = requests.get(url, timeout=45, headers={"User-Agent": "FrontierRAG/1.0 (educational project)"})
    response.raise_for_status()
    target.write_bytes(response.content)
    manifest.append({"filename": filename, "url": url, "attribution": attribution, "bytes": len(response.content)})
    print(f"Downloaded {filename}: {len(response.content):,} bytes")
(ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Saved {len(manifest)} files to {RAW}")
