#!/usr/bin/env bash
set -e

###############################################################################
# COLORED OUTPUT (corporate clean theme)
###############################################################################
CYAN="\033[96m"
GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
MAGENTA="\033[95m"
RESET="\033[0m"

print_step()  { echo -e "${CYAN}$1${RESET}"; }
print_ok()    { echo -e "${GREEN}$1${RESET}"; }
print_warn()  { echo -e "${YELLOW}$1${RESET}"; }
print_err()   { echo -e "${RED}$1${RESET}"; }
print_cmd()   { echo -e "${MAGENTA}$1${RESET}"; }

###############################################################################
print_step "=============================================================="
print_step " Quinten Health — Project Environment Setup"
print_step "=============================================================="
echo ""
###############################################################################


###############################################################################
# STEP 1 — Check for pyenv (install automatically if missing)
###############################################################################
print_step "[1/10] Checking pyenv installation…"

if ! command -v pyenv >/dev/null 2>&1; then
    print_warn "pyenv is not installed."

    print_step "Installing pyenv using the official installer…"
    print_cmd "curl https://pyenv.run | bash"

    curl https://pyenv.run | bash

    print_ok "pyenv installation completed."
else
    print_ok "pyenv is installed."
fi


###############################################################################
# STEP 2 — Configure shell for pyenv + poetry (Option 3: auto reload)
###############################################################################
print_step "[2/10] Checking shell configuration…"

ZSHRC="$HOME/.zshrc"
LOCAL_ZSHRC="$HOME/.zshrc.local"

# If script already reloaded automatically, we skip this section
if [ -n "$QH_SHELL_RELOADED" ]; then
    print_ok "Shell already reloaded — proceeding with setup…"
else
    print_step "Verifying presence of ~/.zshrc.local…"

    # --- Generate ~/.zshrc.local only once
    if [ ! -f "$LOCAL_ZSHRC" ]; then
        print_warn "~/.zshrc.local is missing — creating it…"

        cat > "$LOCAL_ZSHRC" <<EOF
# Auto-generated local zsh configuration for pyenv + poetry
export PYENV_ROOT="\$HOME/.pyenv"
export PATH="\$PYENV_ROOT/bin:\$PATH"
eval "\$(pyenv init --path)"
eval "\$(pyenv init -)"
eval "\$(pyenv virtualenv-init -)"

# poetry integration
export PATH="\$HOME/.local/bin:\$PATH"
EOF

        print_ok "~/.zshrc.local created."
    else
        print_ok "~/.zshrc.local already exists (kept as-is)."
    fi


    # --- Ensure ~/.zshrc loads ~/.zshrc.local
    if ! grep -q "source ~/.zshrc.local" "$ZSHRC"; then
        print_warn "Injecting 'source ~/.zshrc.local' into ~/.zshrc…"
        {
            echo ""
            echo "# Load custom local configuration"
            echo "source ~/.zshrc.local"
        } >> "$ZSHRC"
        print_ok "~/.zshrc updated."
    else
        print_ok "~/.zshrc already loads ~/.zshrc.local."
    fi


    # --- Strong warning before automatic reload
    print_err "IMPORTANT NOTICE:"
    print_warn "Your shell configuration has been modified to use pyenv and poetry together."
    print_warn "Your terminal session will now automatically reload to apply these changes."
    print_warn "This is safe, but will momentarily replace your current shell session."
    echo ""

    print_step "Reloading shell automatically (exec zsh -l)…"
    echo ""

    # Mark reload so we continue after restart
    export QH_SHELL_RELOADED=1

    exec zsh -l -c "QH_SHELL_RELOADED=1 bash $0"
fi

###############################################################################
# STEP 3 — Select Python version (dynamic 3.11.x detection + installation)
###############################################################################
print_step "[3/10] Selecting Python version…"

# The major/minor version we want for this project.
PYTHON_VERSION_PREFIX="3.11"
# Default fallback version to install if none exists
DEFAULT_PYTHON_VERSION="3.11.9"

# Find first installed version starting with 3.11.x
PYTHON_VERSION=$(pyenv versions --bare | grep "^${PYTHON_VERSION_PREFIX}\." | head -n 1)

if [ -z "$PYTHON_VERSION" ]; then
    print_warn "No Python version starting with '${PYTHON_VERSION_PREFIX}' found."

    print_step "Installing Python ${DEFAULT_PYTHON_VERSION} via pyenv…"
    print_cmd "pyenv install ${DEFAULT_PYTHON_VERSION}"

    pyenv install "${DEFAULT_PYTHON_VERSION}"

    PYTHON_VERSION="${DEFAULT_PYTHON_VERSION}"

    print_ok "Python ${PYTHON_VERSION} installed."
else
    print_ok "Found Python version: ${PYTHON_VERSION}"
fi


###############################################################################
# STEP 4 — Create or reuse the project virtualenv
###############################################################################
print_step "[4/10] Checking project virtual environment…"

# Your original behavior: name environment after project folder
PROJECT_NAME=$(basename "$PWD")
ENV_NAME="${PROJECT_NAME}-env"

# Expected path of this environment
ENV_PATH="$HOME/.pyenv/versions/${PYTHON_VERSION}/envs/${ENV_NAME}"

# Create virtualenv if missing
if pyenv virtualenvs --bare | grep -q "^${ENV_NAME}$"; then
    print_ok "Virtual environment '${ENV_NAME}' already exists."
else
    print_warn "Virtual environment '${ENV_NAME}' not found."
    print_step "Creating virtual environment '${ENV_NAME}' using Python ${PYTHON_VERSION}…"

    print_cmd "pyenv virtualenv ${PYTHON_VERSION} ${ENV_NAME}"
    pyenv virtualenv "${PYTHON_VERSION}" "${ENV_NAME}"

    print_ok "Environment '${ENV_NAME}' created."
fi

###############################################################################
# AUTO-DEACTIVATE ANY PREVIOUS ENVIRONMENT (Option 2)
###############################################################################
print_step "Checking if another pyenv environment is active…"

ACTIVE_ENV=$(pyenv version-name || echo "system")

if [ "$ACTIVE_ENV" != "system" ] && [ "$ACTIVE_ENV" != "$ENV_NAME" ]; then
    print_warn "Another environment is currently active: ${ACTIVE_ENV}"
    print_step "Deactivating it to avoid conflicts…"
    print_cmd "pyenv deactivate"
    pyenv deactivate || true
    print_ok "Previous environment deactivated."
else
    print_ok "No conflicting environment active."
fi

###############################################################################
# STEP 5 — Activate virtualenv and set local project version
###############################################################################
print_step "[5/10] Activating environment '${ENV_NAME}'…"

# Make pyenv use this environment for the current project folder
pyenv local "${ENV_NAME}"
print_ok "Set local pyenv environment to '${ENV_NAME}'."

# Check if environment is already active
CURRENT_ENV=$(pyenv version-name)

if [ "$CURRENT_ENV" = "$ENV_NAME" ]; then
    print_ok "Virtual environment '${ENV_NAME}' is already active in this shell."
else
    print_step "Activating virtual environment…"
    print_cmd "pyenv activate ${ENV_NAME}"
    pyenv activate "${ENV_NAME}"
fi

print_ok "Environment '${ENV_NAME}' is active and ready."

###############################################################################
# STEP 6 — Configure Poetry behavior
###############################################################################
print_step "[6/10] Configuring Poetry behavior…"

# Ensure Poetry is installed
if ! command -v poetry >/dev/null 2>&1; then
    print_err "Poetry is not installed."
    print_warn "Please install Poetry with:"
    print_cmd "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

print_ok "Poetry is installed."

print_step "Configuring Poetry to avoid creating virtual environments…"
print_cmd "poetry config virtualenvs.create false --local"
poetry config virtualenvs.create false --local
print_ok "Poetry configured (virtualenvs.create = false)."


###############################################################################
# STEP 7 — Validate Poetry environment
###############################################################################
print_step "[7/10] Validating Poetry environment…"

echo -e "${CYAN}Checking Poetry environment info…${RESET}"
POETRY_ENV_INFO=$(poetry env info 2>/dev/null || true)

if [ -z "$POETRY_ENV_INFO" ]; then
    print_err "Failed to retrieve Poetry environment information!"
    print_warn "This may happen if Poetry cannot detect an active environment."
    print_warn "SOLUTION: Try closing the terminal fully, reopen it, activate the env, then rerun the script."
    exit 1
fi

# Extract venv path ONLY from the Virtualenv section
POETRY_ENV_PATH=$(echo "$POETRY_ENV_INFO" \
    | awk '/Virtualenv/ {flag=1} flag && /Path:/ {print $2; exit}')

# Extract validity
VALID_ENV=$(echo "$POETRY_ENV_INFO" | awk '/Valid:/ {print $2}')

if [ -z "$POETRY_ENV_PATH" ]; then
    print_err "Could not extract Poetry virtualenv path from 'poetry env info'."
    echo "$POETRY_ENV_INFO"
    exit 1
fi

# Guard if fields are empty
if [ -z "$VALID_ENV" ]; then
    print_err "Poetry did not provide a 'Valid:' field."
    echo "$POETRY_ENV_INFO"
    exit 1
fi

# Validation 1: Poetry must report Valid: True
if [ "$VALID_ENV" != "True" ]; then
    print_err "Poetry environment is NOT valid:"
    echo "$POETRY_ENV_INFO"
    print_warn "SOLUTION: close terminal, reopen and rerun this script."
    exit 1
fi

# Validation 2: Check expected path
if [ "$POETRY_ENV_PATH" != "$ENV_PATH" ]; then
    print_err "Poetry environment path mismatch!"
    print_warn "Poetry reports: $POETRY_ENV_PATH"
    print_warn "Expected path:  $ENV_PATH"
    print_err "Poetry is NOT using your pyenv environment."
    print_warn "SOLUTION: remove incorrect env with:"
    print_cmd "poetry env remove python"
    exit 1
fi

print_ok "Poetry environment is valid and correctly configured."


###############################################################################
# STEP 8 — Poetry credentials (setup)
###############################################################################
print_step "[8/10] Checking Poetry PyPI credentials for 'pypi_product'…"

if poetry config --list 2>/dev/null | grep -q 'http-basic.pypi_product'; then
    print_ok "Poetry credentials for 'pypi_product' already configured."
else
    print_warn "Credentials for 'pypi_product' are NOT configured."
    print_step "Setting credentials for private PyPI repository: pypi_product"

    # Prompt the user for PyPI username + password
    # Username shows, password hidden.
    echo -ne "${CYAN}Enter your PyPI username: ${RESET}"
    read PYPI_USERNAME

    echo -ne "${CYAN}Enter your PyPI password: ${RESET}"
    read -s PYPI_PASSWORD
    echo ""

    print_step "Configuring Poetry with provided credentials…"
    poetry config http-basic.pypi_product "$PYPI_USERNAME" "$PYPI_PASSWORD"

    print_ok "Poetry has been successfully configured with 'pypi_product' credentials."
fi

###############################################################################
# STEP 9 — Install project dependencies with Poetry
###############################################################################
print_step "[9/10] Installing project dependencies with Poetry…"

# Ensure a lockfile exists or refresh it
if [ ! -f "poetry.lock" ]; then
    print_warn "No poetry.lock found — generating one with 'poetry lock'…"
    print_cmd "poetry lock"
    poetry lock
else
    print_step "Refreshing lockfile with 'poetry lock'…"
    print_cmd "poetry lock"
    poetry lock
fi

print_step "Installing dependencies using 'poetry install'…"
print_cmd "poetry install"
poetry install

print_ok "Poetry dependencies installed successfully."


###############################################################################
# STEP 10 — Final success message
###############################################################################
print_step "[10/10] Setup Complete!"
print_ok "Your Python + pyenv + Poetry environment is fully configured."
print_ok "Virtual environment: ${ENV_NAME}"
print_ok "Python version: ${PYTHON_VERSION}"
print_ok "Environment path: ${ENV_PATH}"

echo ""
print_step "=============================================================="
print_step " 🎉 Your project environment is READY — Happy coding! 🎉"
print_step "=============================================================="
echo ""
