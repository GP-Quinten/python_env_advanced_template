# P00 Template Project

This is a structured template for a Snakemake-based project with environment management via **pyenv** and **Poetry**.

## 📂 Project Structure

```
project/
│── data/                     # Contains generated and processed datasets
│   ├── R01_generate_dataset/  # Raw dataset
│   ├── R02_clean_dataset/     # Cleaned dataset
│── notebooks/                 # Jupyter notebooks for analysis
│   ├── N01_hello_world.ipynb  # Example notebook
│── src/                       # Source code and scripts
│   ├── scripts/
│       ├── S01_generate_dataset.py  # Dataset generation script
│       ├── S02_clean_dataset.py     # Dataset cleaning script
│   ├── p00_template/           # Python package
│       ├── __init__.py
│       ├── hello_world.py
│── Snakefile                   # Snakemake workflow definition
│── setup_env.sh                # Script to set up the environment
│── pyproject.toml               # Poetry dependencies
│── README.md                    # This file
```

## 🚀 Setup

1. **Set up the environment**  
   ```bash
   ./setup_env.sh
   ```

2. **Pull data from DVC**
   ```bash
   dvc pull
   ```

3. **Run Snakemake workflow**  
   ```bash
   snakemake R01_generate_dataset
   ```

4. **Open Jupyter notebook**  
   ```bash
   jupyter notebook notebooks/N01_hello_world.ipynb
   ```

---

This template ensures a reproducible and structured workflow for data processing and analysis using Snakemake and Jupyter notebooks. 🚀

