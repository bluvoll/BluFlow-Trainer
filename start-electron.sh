#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
command -v node >/dev/null 2>&1 || { printf '%s\n' 'Node.js is required. Run setup-electron.sh first.' >&2; exit 1; }
exec node scripts/electron.cjs start "$@"
