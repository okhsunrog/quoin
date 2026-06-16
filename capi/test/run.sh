#!/usr/bin/env bash
# Build the static C library and the C round-trip test, then run it.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build --release --manifest-path Cargo.toml >&2

# Locate the static lib (the build dir may be redirected). `-print -quit`
# stops at the first match without a pipe (avoids SIGPIPE under pipefail).
SEARCH_DIRS=(. /home/okhsunrog/tmp_zfs/rust_build)
[ -n "${CARGO_TARGET_DIR:-}" ] && SEARCH_DIRS+=("$CARGO_TARGET_DIR")
LIB=$(find "${SEARCH_DIRS[@]}" -name 'libquoin_capi.a' -print -quit 2>/dev/null)
[ -n "$LIB" ] || { echo "static lib not found" >&2; exit 1; }
echo "linking against $LIB" >&2

CC=${CC:-clang}
"$CC" -O2 -Iinclude test/roundtrip.c "$LIB" \
  -lpthread -ldl -lm -o test/roundtrip_c

exec ./test/roundtrip_c
