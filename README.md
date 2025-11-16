# Norm Reference: Dataset and Zero-shot Norm Evaluation

⚖️ This repository contains datasets, scripts, and analysis code to evaluate large language models (LLMs) on normative object reference tasks. The experiments evaluate models’ ability to identify objects given a referring expression and a scene context while reasoning about social norms (serving, cleaning, cooking, etc.).

The code and datasets produced here were used for the research described in the paper `Where Norms and References Collide: Evaluating LLMs on Normative Reasoning`.

---

## Contents

- `dataset/` : The dataset folder. Key datasets:
	- `NBRR_dataset.csv` — original dataset used in experiments (contains scene prompts, prolog code, target object, etc.)
	- `NBRR_dataset_extended_text.csv` — extended dataset with textual augmentations
	- `NBRR_dataset_extended_prolog.csv` — extended dataset with prolog descriptions
- `setup_functions.py` : Utility functions, dataset loader, dspy signature classes for several evaluation modes, LM setup helpers, and a simple accuracy helper.
- `run.py` : Main script to run zero-shot evaluation on the extended dataset using the dspy framework and either Ollama or OpenAI LMs.
- `run_original.py` : Run the original dataset (by norm group), compute category-wise and overall accuracy, and export responses.
- `object_analysis.py` : Dataset statistics (object counts per scene/task, attribute counts, etc.)
- `combine_datasets.py` : Merge multiple CSV files in a dataset folder into a single `combined_dataset.csv`.
- `statistical_analysis.py` : Builds Spearman correlation matrix and visualizations for the model results across norms.
- `results/` : Sample outputs and metrics (CSV files with accuracies, responses, etc.).
- `requirements.txt` : Python packages required.
- Paper: "Where Norms and References Collide: Evaluating LLMs on Normative Reasoning" (research paper describing dataset and experiments).

---

## Quick Start

Prerequisites:

- Python 3.10+ (3.11 recommended where convenient)
- `pip` and `virtualenv` (or your chosen env manager)
- Either Ollama running locally (for `ollama_chat/*` models) or a valid OpenAI API key (for OpenAI models). The default `setup_functions.lm_setup` uses an Ollama API base at `http://localhost:11434`.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- `dspy` is used to define model signatures and to call the LM. See `setup_functions.py` for the signature definitions (ZeroShot, ZeroShotICL, ZeroShotNoProlog, ZeroShotNoPrologICL, ZeroShotReasoning, etc.).
- The scripts assume a dataset structure compatible with `NBRR_dataset.csv` (columns: `prompt`, `prolog`, `all_objects`, `target_object`, `Source File`, ...).

---

## End-to-end run examples

1) Running the extended dataset (using `phi4-mini` / Ollama by default):

```bash
# ensure Ollama is running locally (or switch to OpenAI below)
python run.py
```

By default `run.py`:
- uses `setup_functions.lm_setup()` to call a local Ollama server (`api_base='http://localhost:11434'`), `model_name = "phi4-mini"`.
- reads `dataset/NBRR_dataset.csv` or `dataset/NBRR_dataset_extended_text.csv` depending on the configured path.
- writes model predictions and a per-source accuracy CSV into the `results/` folder (see the printed output in the console).

2) Using OpenAI (run_original):

```bash
# update your API key in run_original.py (or modify code to read from env var)
python run_original.py
```

`run_original.py` runs a simple category-wise accuracy and prints results for each `Norm` (e.g. Norm 1, Norm 2, etc.). Configure `model_name` and `api_key` variables for your environment.

3) Combine CSV files into a single dataset (if you have a folder of CSVs):

```bash
python combine_datasets.py
```

This script calls `setup_functions.list_from_folder()` and `combine_csvs()` and writes `combined_dataset.csv` in `dataset/combined` (you can update the path in the script to fit your folder structure).

---

## Files & Usage Notes

- `setup_functions.py`: Contains helper functions and data signatures. `ZeroShot`, `ZeroShotOriginal`, and other signature classes are used by `dspy` for structured LLM output.
	- LM Setup:
		- `lm_setup(model_name)` → uses Ollama local server.
		- `lm_setup_openai(model_name, api_key)` → uses OpenAI backend.
	- `accuracy(responses, targets)` computes accuracy across responses.

- `run.py`
	- Intended for the extended dataset (NBRR extended). Default: `model_name = "phi4-mini"`.
	- By default, uses `ZeroShot` signature, which includes `scene`, `prolog_code` inputs and expects `target_object` output.
	- To run without Prolog, uncomment the `ZeroShotNoProlog` lines and comment the `ZeroShot` ones.
	- Output: `results/responses_norm_extended_<model_name>.csv`, `results/accuracies_norm_extended<model_name>.csv` (accuracy per source and overall).

- `run_original.py`
	- Uses `ZeroShotOriginal` signature which only accepts `scene` and returns `target_object`.
	- Designed to run the original dataset `dataset/original_dataset_by_norm_group.csv` and to print category-wise results.
	- Uses `setup_functions.lm_setup_openai()` by default — set `api_key` or modify to use `ollama`.

- `object_analysis.py`:
	- Helpful scripts to compute dataset stats: object count, average objects per row, attribute counts.

- `statistical_analysis.py`:
	- Uses numeric/per-model results to compute Spearman correlations across norms and to produce a heatmap (saved as `spearman_correlation_matrix.png`).

---

## Datasets & Columns (short)

- `prompt` — Natural language description of the scene (with the referring expression)
- `name_pair` — Example participants in the scene
- `location` — Location: `library`, `kitchen`, etc.
- `task` — Task: `tidying`, `serving`, `cleaning`, etc.
- `prolog` — The Prolog representation of the scene (objects and properties)
- `referring_expression` — The referring phrase used in the prompt
- `target_object` — The ground-truth object (e.g. `obj3`) that should be picked
- `all_objects` — The full list of objects in the scene (used as options)
- `target_description` — Textual description of the target object

You can open `dataset/NBRR_dataset.csv` and sample entries to better understand label and format.

---

## Reproducibility & Best Practices

- Use a deterministic sampling setup to reduce stochasticity: `dspy.configure(lm=..., temperature=0)` is used by default in `setup_functions`.
- If you rely on OpenAI's API, keep your `api_key` secure: set it in an environment variable or use a secure secrets manager; `run_original.py` currently stores a placeholder for this key.
- Ensure the model server (if using Ollama) is accessible at `http://localhost:11434` (or change the `api_base` in `lm_setup`).
- For large-scale runs, pipe outputs to disk and/or sample the dataset to speed iterations.

---

## Analysis & Visualization

- `object_analysis.py` helps check dataset properties (e.g., average object counts).
- `statistical_analysis.py` contains pre-computed performance arrays and uses them to compute Spearman correlations and generate a heatmap (`spearman_correlation_matrix.png`). The arrays here appear to be pre-populated results; adapt the script to point to CSV results for automated runs.

---

## Adding a New Model / Dataset

1. Add your dataset to `dataset/` ensuring expected columns (`prompt`, `prolog`, `all_objects`, `target_object`, `Source File`, etc.)
2. Modify `run.py` or `run_original.py` to set `model_name` and choose `lm_setup` vs. `lm_setup_openai`.
3. Optionally modify the `Signature` used — `ZeroShot`, `ZeroShotNoProlog`, etc.
4. Run the corresponding script and review `results/` for generated CSVs.

---

## (Optional) Example: Run a quick test

```bash
# run object analysis quick stats
python object_analysis.py

# combine dataset CSVs from a folder (if needed)
python combine_datasets.py

# quick run using a small subset (update run.py to use a subset or split dataset)
python run.py
```

---

## Troubleshooting

- If you get `dspy` signature errors, update the `run.py` script to ensure you’re using the right `dspy.Predict(setup_functions.<Signature>)` with the correct input fields.
- If Prolog code is missing from a dataset, use the `ZeroShotNoProlog` signature instead of `ZeroShot`.
- If model responses don’t adhere to the output signature (dspy errors), use the retry logic included in `run.py` and `run_original.py` (or wrap LM calls with additional guard clauses.)

---

## Results and Outputs

Output files are saved to `results/`. Typical files include:
- `responses_norm_extended_<model>.csv` — model predictions (one row per dataset entry)
- `accuracies_norm_extended<model>.csv` — per-source and overall accuracy
- Other accuracy CSVs and responses (`accuracies_*`, `responses_*`) used to compile the figures for the paper

---

## Citation

If you use these datasets/code in your research, please cite the paper and provide attribution: Where Norms and References Collide: Evaluating LLMs on Normative Reasoning. See the paper for details on dataset creation and experiments.

