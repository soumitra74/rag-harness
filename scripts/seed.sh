#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${API_BASE_URL:-http://localhost:8098}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# cd into the repo root and reference files by relative name: some curl
# builds (e.g. Windows/git-bash mingw curl) fail to read an absolute
# POSIX-style path (/d/...) passed inside a -F "file=@..." value.
cd "$REPO_ROOT"

# corpus/ is the canonical source — every PDF placed there gets ingested.
# ingest_pdf() is idempotent per filename+config, so re-running this script
# is safe and won't create duplicate documents.
for pdf in corpus/*.pdf
do
  echo "Uploading $pdf..."
  curl -sf -X POST "$API_BASE_URL/api/documents" \
    -F "file=@${pdf};type=application/pdf"
  echo
done
