#!/usr/bin/env bash
set -e

# Auto-deactivate any active pyenv environment before starting setup
ACTIVE_ENV=$(pyenv version-name || echo "system")

if [ "$ACTIVE_ENV" != "system" ]; then
    echo "Deactivating active environment: $ACTIVE_ENV"
    pyenv deactivate || true
fi

# Now run the shared setup script
bash ../../etc/setup_env.sh