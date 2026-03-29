#!/usr/bin/env bash
# Download open PDFs from official sources (run from repo root).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/corpus/raw"
mkdir -p "$OUT"

UA="Mozilla/5.0 (compatible; rag-defence-poc corpus script)"

echo "==> Defensie Strategie Data Science en AI 2023-2027 (Open Overheid)"
curl -fsSL -A "$UA" -o "$OUT/defensie-strategie-ds-ai-2023-2027.pdf" \
  "https://open.overheid.nl/documenten/d49f42ca-181b-4e2f-9986-b412de40f2f5/file"

echo "==> Overheidsbrede handreiking generatieve AI (Open Overheid)"
curl -fsSL -A "$UA" -o "$OUT/overheidsbrede-handreiking-generatieve-ai.pdf" \
  "https://open.overheid.nl/documenten/9c273b71-cebb-4e11-b06f-fa20f7b4b90e/file"

echo "Done. Files in: $OUT"
ls -lh "$OUT"
