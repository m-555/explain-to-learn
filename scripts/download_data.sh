#!/usr/bin/env bash
# Download the BEA-2019 versions of W&I+LOCNESS and FCE, and the GEE data repository,
# into data/raw/. Files that are already present are skipped.
#
# Usage: bash scripts/download_data.sh --accept-licence
set -euo pipefail

BEA_URL="https://www.cl.cam.ac.uk/research/nl/bea2019st/data"
GEE_URL="https://github.com/Yixiao-Song/GEE-with-LLMs"
RAW_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/raw"

# Archive name, SHA-256 checksum and the directory it extracts to.
ARCHIVES=(
  "wi+locness_v2.1.bea19.tar.gz d5cbf68cda3da0c3af69dd672614d07287bfe996b87da0c75051d5349d76c666 wi+locness"
  "fce_v2.1.bea19.tar.gz c574c1cdba6d3ab5a87280f180133cdcf0609848f5dc87cfa2c3f4b0c07ec67e fce"
)

cat <<'NOTICE'
Licence notice
--------------
W&I+LOCNESS and FCE are released for non-commercial research and educational use only.
Each archive contains its licence (licence.wi.txt, license.locness.txt, licence.txt).
By downloading them you enter into these licence agreements. In short:
  - non-commercial research and educational purposes only;
  - cite the papers named in the licences in any publication;
  - W&I and FCE: excerpts of fewer than 100 words may be published;
  - LOCNESS: no part may be passed to a third party without permission from CECL,
    UCLouvain, and publications must credit CECL.
Read the terms first: https://www.cl.cam.ac.uk/research/nl/bea2019st/
NOTICE

if [[ "${1:-}" != "--accept-licence" ]]; then
  echo
  echo "Re-run with --accept-licence once you have read and accepted the licence terms."
  exit 1
fi

mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

for entry in "${ARCHIVES[@]}"; do
  read -r name checksum folder <<< "$entry"
  if [[ -f "$name" ]]; then
    echo "Skipping download, $name already present."
  else
    echo "Downloading $name"
    curl -fL --retry 3 -o "$name.part" "$BEA_URL/$name"
    mv "$name.part" "$name"
  fi
  echo "$checksum  $name" | sha256sum --check --quiet
  if [[ -d "$folder" ]]; then
    echo "Skipping extraction, $folder/ already present."
  else
    tar -xzf "$name"
  fi
done

if [[ -d gee/.git ]]; then
  echo "Skipping clone, gee/ already present."
else
  git clone --depth 1 "$GEE_URL" gee
fi

echo "Data is in $RAW_DIR"
