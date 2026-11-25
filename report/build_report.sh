#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

pandoc report.md \
  --from markdown \
  --pdf-engine=xelatex \
  -V papersize:a4 \
  -V fontsize=10pt \
  -V geometry:margin=2cm \
  -V linestretch=1.0 \
  -V colorlinks=true \
  -V urlcolor=blue \
  -o report.pdf
