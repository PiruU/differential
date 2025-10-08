#!/usr/bin/env bash

set -e
set -u

echo "Cleaning project"
EXEC="-exec rm -rf {} +"
find . -type d -name "__pycache__" -print ${EXEC}
find . -type d -name ".pytest_cache" -print ${EXEC}
find . -name "*.egg-info" -print ${EXEC}
echo "Done."
