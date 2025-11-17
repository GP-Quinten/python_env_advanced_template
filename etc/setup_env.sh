#!/usr/bin/env bash
set -e

# -----------------------------------------------------------------------------
# setup_env.sh — Environment setup helper (clearer comments for juniors)
# -----------------------------------------------------------------------------
# Purpose (simple):
#   1) Ensure pyenv is installed and a compatible Python 3.11.x exists.
#   2) Create or reuse a pyenv virtualenv named <project>-env.
#   3) Activate that virtualenv and ensure Poetry is using it.
#   4) Configure Poetry PyPI credentials and install dependencies.
#
# Quick notes for juniors:
#   - Run this script from the project root (it writes .python-version).
#   - Make sure pyenv and Poetry are installed before running this script.
#   - The script is intentionally explicit so you can see each step clearly.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Check for pyenv
# -----------------------------------------------------------------------------
# Check if pyenv is installed. If not, show an actionable message and exit.
if ! command -v pyenv >/dev/null; then
    echo "pyenv is not installed. Please install pyenv first." >&2
    echo "SOLUTION: Following instructions:" >&2
    echo "  $ curl https://pyenv.run | bash" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Initialize pyenv in this shell
# -----------------------------------------------------------------------------
# Load pyenv and the virtualenv commands into this shell session.
# These lines set up shell integration so pyenv and pyenv-virtualenv work here.
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# -----------------------------------------------------------------------------
# Advise user to update ~/.zshrc for Poetry and pyenv shell integration
# -----------------------------------------------------------------------------
echo ""
echo "IMPORTANT: To ensure Poetry and pyenv work correctly in your shell, please execute the following commands in your terminal:"
echo ""
echo "echo '# pyenv shell integration' > ~/.zshrc.local"
echo "echo 'export PYENV_ROOT=\"\$HOME/.pyenv\"' >> ~/.zshrc.local"
echo "echo 'export PATH=\"\$PYENV_ROOT/bin:\$PATH\"' >> ~/.zshrc.local"
echo "echo 'eval \"\$(pyenv init --path)\"' >> ~/.zshrc.local"
echo "echo 'eval \"\$(pyenv init -)\"' >> ~/.zshrc.local"
echo "echo 'eval \"\$(pyenv virtualenv-init -)\"' >> ~/.zshrc.local"
echo "echo '' >> ~/.zshrc.local"
echo "echo '# Poetry shell integration' >> ~/.zshrc.local"
echo "echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc.local"
echo ""
echo "echo '' >> ~/.zshrc"
echo "echo '# Load custom local configuration' >> ~/.zshrc"
echo "echo 'source ~/.zshrc.local' >> ~/.zshrc"
echo ""
echo "This ensures 'pyenv' and 'poetry' commands are available in every new terminal session."
echo ""
echo "IMPORTANT!!! After updating ~/.zshrc, restart your terminal or run 'source ~/.zshrc' to apply the changes."

# -----------------------------------------------------------------------------
# Select Python version
# -----------------------------------------------------------------------------
# Desired Python version prefix we want to use for this project.
PYTHON_VERSION_PREFIX="3.11"

# Pick the first installed Python version that starts with the prefix.
PYTHON_VERSION=$(pyenv versions --bare | grep "^${PYTHON_VERSION_PREFIX}" | head -n 1)
if [ -z "$PYTHON_VERSION" ]; then
    echo "No Python version starting with ${PYTHON_VERSION_PREFIX} is installed in pyenv. Please install it and try again." >&2
    echo "SOLUTION: execute following command:" >&2
    echo "  $ pyenv install ${PYTHON_VERSION_PREFIX}" >&2
    echo "If it fails, ensure you have the necessary build dependencies installed for your OS." >&2
    echo "https://github.com/pyenv/pyenv/wiki#suggested-build-environment" >&2
    echo "Contact IT if you need help." >&2
    exit 1
fi

echo "Using Python version: ${PYTHON_VERSION}"

# -----------------------------------------------------------------------------
# Create or reuse the project virtualenv
# -----------------------------------------------------------------------------
# Name the virtualenv after the project directory to make it obvious.
PROJECT_NAME=$(basename "$PWD")
ENV_NAME="${PROJECT_NAME}-env"
ENV_PATH="$HOME/.pyenv/versions/${PYTHON_VERSION}/envs/${ENV_NAME}"

# Create the virtualenv if it does not already exist.
if pyenv virtualenvs --bare | grep -q "^${ENV_NAME}$"; then
    echo "Virtual environment ${ENV_NAME} already exists."
else
    echo "Creating virtual environment ${ENV_NAME} using Python ${PYTHON_VERSION}..."
    pyenv virtualenv "${PYTHON_VERSION}" "${ENV_NAME}"
fi

# -----------------------------------------------------------------------------
# Activate virtualenv and set local project version
# -----------------------------------------------------------------------------
# Make this directory use the named virtualenv by default (.python-version file).
pyenv local "${ENV_NAME}"
echo "Set local pyenv environment: ${ENV_NAME}"

# If the environment is not active in this shell, activate it now.
CURRENT_ENV=$(pyenv version-name)
if [ "$CURRENT_ENV" = "$ENV_NAME" ]; then
    echo "Virtual environment ${ENV_NAME} is already activated."
else
    echo "Activating virtual environment: ${ENV_NAME}"
    pyenv activate "${ENV_NAME}"
fi

# -----------------------------------------------------------------------------
# Configure Poetry behavior
# -----------------------------------------------------------------------------
# Ensure Poetry will not create its own virtual env and instead use the current one.
if command -v poetry >/dev/null; then
    echo "Configuring Poetry to not create virtual environments (virtualenvs.create = false)..."
    poetry config virtualenvs.create false --local
else
    echo "Warning: Poetry is not installed; skipping poetry virtualenvs.create configuration."
    echo "SOLUTION: Please install Poetry first by executing:" >&2
    echo "  $ curl -sSL https://install.python-poetry.org | python3 -" >&2
    echo "Then ensure \$HOME/.local/bin is in your PATH. Update your ~/.zshrc as needed. And restart your terminal." >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Validate Poetry environment
# -----------------------------------------------------------------------------
# Validate Poetry's environment information so we can compare paths and validity.
echo "Checking Poetry environment..."
POETRY_ENV_INFO=$(poetry env info 2>/dev/null)

# Extract the first Path: line (the virtualenv path) and the Valid: flag from Poetry output.
POETRY_ENV_PATH=$(echo "$POETRY_ENV_INFO" | awk '/Path:/ {print $2; exit}')
VALID_ENV=$(echo "$POETRY_ENV_INFO" | awk '/Valid:/ {print $2}')

# If Poetry reports the environment is not valid, show details and exit.
if [[ "$VALID_ENV" != "True" ]]; then
    echo "Error: Poetry reports that the environment is not valid." >&2
    echo "$POETRY_ENV_INFO" >&2
    echo "SOLUTION: please simply rerun this script." >&2
    exit 1
fi

# If Poetry's reported venv path does not match the expected pyenv path, show details and exit.
if [[ "$POETRY_ENV_PATH" != "$ENV_PATH" ]]; then
    echo "Error: Poetry environment path ($POETRY_ENV_PATH) does not match expected path ($ENV_PATH)." >&2
    echo "$POETRY_ENV_INFO" >&2
    exit 1
fi

echo "Verified correct Poetry virtual environment: $POETRY_ENV_PATH"

# Check if Poetry is installed (again) and exit if missing.
if ! command -v poetry >/dev/null; then
    echo "Poetry is not installed. Please install Poetry first." >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Poetry credentials (setup)
# -----------------------------------------------------------------------------
# Configure Poetry PyPI credentials for 'pypi_product' if not already set.
echo "Checking Poetry PyPI credentials for 'pypi_product'..."
if poetry config --list 2>/dev/null | grep -q 'http-basic.pypi_product'; then
    echo "Poetry credentials for 'pypi_product' already configured."
else
    # Prompt the user for PyPI username and password and set them in Poetry config.
    read -p "Enter your PyPI username: " PYPI_USERNAME
    read -s -p "Enter your PyPI password: " PYPI_PASSWORD
    echo ""
    poetry config http-basic.pypi_product "$PYPI_USERNAME" "$PYPI_PASSWORD"
    echo "✅ Poetry has been configured with credentials for 'pypi_product'."
fi
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Install project dependencies with Poetry
# -----------------------------------------------------------------------------
# === Poetry install ===
# Ensure a lockfile exists and refresh it, then install dependencies via Poetry.
if [ ! -f "poetry.lock" ]; then
    echo "No poetry.lock found — generating one with 'poetry lock'..."
    poetry lock
else
    echo "Running 'poetry lock' to ensure lockfile is up-to-date..."
    poetry lock
fi

# Install project dependencies via Poetry.
echo "Running 'poetry install'..."
poetry install
# === End Poetry install ===

echo "Setup complete."
