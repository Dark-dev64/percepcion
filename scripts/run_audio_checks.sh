#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
AUDIO_TEST_ENV="${AUDIO_TEST_ENV:-$PWD/.venv-audio}"
"$AUDIO_TEST_ENV/bin/python" -m pytest tests -v
