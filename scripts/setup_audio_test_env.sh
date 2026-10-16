#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
AUDIO_TEST_ENV="${AUDIO_TEST_ENV:-$PWD/.venv-audio}"
python3 -m venv --without-pip "$AUDIO_TEST_ENV"
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$AUDIO_TEST_ENV/get-pip.py"
"$AUDIO_TEST_ENV/bin/python" "$AUDIO_TEST_ENV/get-pip.py"
"$AUDIO_TEST_ENV/bin/python" -m pip install 'numpy<2' 'scipy<1.18' PyYAML Flask opencv-python-headless==4.10.0.84 -r requirements-audio.txt
