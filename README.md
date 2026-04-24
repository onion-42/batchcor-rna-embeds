# BatchCorrect_RNA_embeds

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project is about evaluating batch correction influence on downstream task performance of clinical endpoint classifiers that were based on RNA embeddings of Foundation RNA models. BG Internship 2026(Group 7).

Сделать окружение:
```python
python3.11 -m venv <YOUR_VENV_FOLDER>/batcor_env
source <YOUR_VENV_FOLDER>/batcor_env/bin/activate
pip install -e  . #from this repo folder!!
python -m ipykernel install \
  --user \
  --name=batcor_env \
  --display-name "Python3.11 (batcor_env)"
  ```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         batchcor_rna_emb and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
│
└── batchcor_rna_emb   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes batchcor_rna_emb a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

