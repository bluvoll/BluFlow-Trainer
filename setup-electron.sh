#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
command -v node >/dev/null 2>&1 || { printf '%s\n' 'Node.js 22.12 or newer is required.' >&2; exit 1; }
exec node scripts/electron.cjs setup "$@"
