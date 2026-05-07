# Pareto-Optimal Fair Meta-Learning Recommender System

This repository implements a fairness-aware multi-objective optimization framework for recommender systems based on Heterogeneous Information Networks (HINs). It jointly balances four objectives using the Multiple Gradient Descent Algorithm (MGDA):
1. **Accuracy**
2. **User Fairness** (e.g., Male vs. Female user groups)
3. **Item Exposure Fairness** (Popular vs. Long-tail items)
4. **Path Fairness** (Balancing the exposure of meta-paths)

## Installation

```bash
pip install -r requirements.txt
```

## Configuration
You can configure the model, hyperparameters, and the dataset by editing `config.json` at the root directory.

To change the dataset, simply update the `dataset` and `data_dir` fields in `config.json`. For example:
```json
{
    "dataset": "yelp",
    "data_dir": "data/yelp",
    ...
}
```

## Usage

### Local Environment
1. Install dependencies: `pip install -r requirements.txt`
2. Configure your `config.json`.
3. Run training: `python trainer.py`. Model weights will be saved in the `models/` directory.
4. Evaluate the Pareto model: `python evaluator.py --model pareto --config config.json`
5. Evaluate the Original model: `python evaluator.py --model original --config config.json`
   (Results will be saved in `results_pareto.json` or `results_original.json`)

### Kaggle Environment
If running on Kaggle, follow these steps:
1. Upload this repository as a Dataset or clone it directly in your Notebook.
2. Ensure you have enabled GPU acceleration in the Notebook settings.
3. Install dependencies by running: `!pip install -r requirements.txt`
4. Update `config.json` to point to the correct data paths (e.g., `/kaggle/input/your-dataset/`).
5. Run the training script via a cell:
   ```python
   !python trainer.py
   ```
6. Run the evaluator:
   ```python
   !python evaluator.py --model pareto
   ```
7. Output files like `results_pareto.json` can be downloaded directly from the Kaggle working directory (`/kaggle/working/`).
