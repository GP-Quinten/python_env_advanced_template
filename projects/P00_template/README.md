# P00 Template Project

This is a structured template for an environment management via **pyenv** and **Poetry**.

## 📂 Project Structure

```
project/
│── notebooks/                 # Jupyter notebooks for analysis
│   ├── N00_testing.ipynb  # Example notebook
│   ├── p00_template/           # Python package
│       ├── __init__.py
│       ├── hello_world.py
│── setup_env.sh                # Script to set up the environment
│── pyproject.toml               # Poetry dependencies
│── README.md                    # This file
```

## 🚀 Setup

**Set up the environment**  
   ```bash
   ./setup_env.sh
   ```
This will run the file ../../etc/setup_env.sh, create your virtual environment and the right dependencies using pyenv + poetry. It will warn you if there is an error.

**Private package dependency**:

This project uses an internal package hosted on Quintens' private PyPI: `llm_backends` (see https://gitlab.par.quinten.io/qlab/llm_backends_package). The `etc/setup_env.sh` script helps set the Poetry credentials required to access that private registry; ensure you have the necessary access and credentials before running `poetry install`.

Run the notebook N00_testing.inpyb to validate your installation of your new environment.
