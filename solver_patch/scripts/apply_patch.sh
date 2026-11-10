#!/usr/bin/env bash
#
# Copy the upstream I2VIS tree, verify its hashes, apply one GeoSim branch
# patch, and install that branch's case configuration.
#
# Usage:
#   bash apply_patch.sh plume1 /path/to/upstream/codes/codes [destination]
#   bash apply_patch.sh plume2 /path/to/upstream/codes/codes [destination]
#
# The destination directory must not already exist.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH="${1:-}"
UPSTREAM="${2:-}"
DEST="${3:-i2vis-${BRANCH}}"

if [[ "$BRANCH" != "plume1" && "$BRANCH" != "plume2" ]]; then
    echo "usage: $0 <plume1|plume2> /path/to/upstream/codes/codes [destination]" >&2
    exit 2
fi

if [[ -z "$UPSTREAM" ]]; then
    echo "usage: $0 <plume1|plume2> /path/to/upstream/codes/codes [destination]" >&2
    exit 2
fi

if [[ ! -d "$UPSTREAM" ]]; then
    echo "upstream directory not found: $UPSTREAM" >&2
    exit 2
fi

if [[ -e "$DEST" ]]; then
    echo "destination already exists: $DEST" >&2
    echo "choose a new directory; this script does not overwrite." >&2
    exit 2
fi

cp -a "$UPSTREAM" "$DEST"
cd "$DEST"

if ! sha256sum -c "$HERE/UPSTREAM.sha256"; then
    echo "upstream hashes do not match the patch base." >&2
    echo "obtain the matching OSF archive or inspect UPSTREAM.md." >&2
    exit 3
fi

patch -p1 < "$HERE/patch/i2vis-${BRANCH}.patch"

if ! sha256sum -c "$HERE/PATCHED.${BRANCH}.sha256"; then
    echo "patched file hashes do not match the expected result." >&2
    exit 4
fi

cp "$HERE/config/${BRANCH}/Makefile" "$DEST/Makefile"
cp "$HERE/config/${BRANCH}/init.t3c" "$DEST/init.t3c"
cp "$HERE/config/${BRANCH}/mode.t3c" "$DEST/mode.t3c"
mkdir -p "$DEST/database"

cat <<EOF
Patch applied: ${BRANCH}

Next steps:
  1. Populate database/ with the thermodynamic tables listed in
     database/README.md.
  2. Build with: make
  3. Run: ./in2fast; ./i2fast; ./i2jslab

The upstream source itself is not redistributed by this package.
EOF
