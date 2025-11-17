# More Europa ML Experiments

This repository contains machine learning experiments and shared resources for the More Europa project. It provides a structured approach to running reproducible ML workflows, managing shared datasets, and leveraging common libraries and scripts.

- Document with the main information about the project - [More Europa_ ML upgrade](https://quintenit-my.sharepoint.com/:w:/r/personal/b_kopin_quinten-health_com/_layouts/15/Doc.aspx?sourcedoc=%7BE158189A-1F34-4CCE-AD25-0CE0F0C2FE82%7D&file=More%20Europa_%20ML%20upgrade.docx&action=default&mobileredirect=true)

## Structure Overview

### `datasets`

- Contains datasets shared across multiple experiments.
- Managed using [Data Version Control (DVC)](https://dvc.org/) for efficient version tracking and reproducibility.

### `projects`

- Independent projects focused on specific machine learning topics.
- Each project is self-contained and includes:
  - **Untracked Data**: Specific datasets unique to the project (not managed by DVC).
  - **Reproducible Workflows**: Defined using [Snakemake](https://snakemake.readthedocs.io/en/stable/) for automated, scalable, and reproducible analysis.
  - Snakemake guidelines (internal): https://quinten-france.atlassian.net/wiki/spaces/MOR/pages/1740734467/Snakemake+Guidelines

**Example Project (`P00_template`):**

```
projects/
└── P00_template/
    ├── README.md
    ├── Snakefile
    ├── data
    │   ├── D00_random_data
    │   ├── R01_generate_dataset
    │   └── R02_clean_dataset
    ├── notebooks
    │   └── N01_hello_word.ipynb
    ├── poetry.lock
    ├── pyproject.toml
    └── src
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

Private package dependency:

This repository uses an internal package hosted on Quintens' private PyPI: `llm_backends` (see https://gitlab.par.quinten.io/qlab/llm_backends_package). The `etc/setup_env.sh` script helps set the Poetry credentials required to access that private registry; ensure you have the necessary access and credentials before running `poetry install`.

## Testing

Tests are available under the `tests/` directory and can be run with:

```bash
pytest
```

---

Feel free to contribute by improving the existing codebase or adding new projects related to the More Europa ML initiative.

