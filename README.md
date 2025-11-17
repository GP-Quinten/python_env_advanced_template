# Python Environment Advanced - tutorial

This repository is a template for advanced python environment management. 

## Structure Overview

### `projects`

- Independent projects focused on specific (machine learning) topics.
- Each project is self-contained and includes code specific to each other.

**Example Project (`P00_template`):**

```
projects/
└── P00_template/
    ├── README.md
    ├── notebooks
    │   └── N00_testing.ipynb
    ├── poetry.lock
    ├── pyproject.toml
    └── src/
    │   └── p00_template/
```

For more details, see the [README](projects/P00_template/README.md) of the `P00_template` project.

### Shared Code

- Shared code is available at:
  - `src/more_europa`: Common modules, helpers, and settings.
  - `src/llm_inference`: Modules for inference using language models.

## Quickstart

Run the helper script at `etc/setup_env.sh` from the project root to prepare a local development environment. In short, the script will:

- Ensure pyenv is available and select a Python 3.11.x interpreter.
- Create (or reuse) a pyenv virtualenv named <project>-env and write a `.python-version` pointing to it.
- Activate the virtualenv for the current shell and configure Poetry to use the existing venv (sets `virtualenvs.create = false`).
- Validate that Poetry's environment matches the pyenv venv and will exit with an actionable error if not.
- Prompt to configure Poetry HTTP basic credentials for the private registry entry `pypi_product` (used for internal packages).
- Run `poetry lock` and `poetry install` to install project dependencies.

Notes:
- The script echoes a suggested `~/.zshrc` snippet for pyenv and Poetry integration — add that to your shell config to ensure the tools are available in new terminals.
- The virtualenv name is derived from the project directory, so you can reuse this script for forks or other projects by running it from the project's root directory.

