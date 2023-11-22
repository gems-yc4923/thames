# ModelSelector

The `ModelSelector` class is a comprehensive tool designed to simplify the model selection process in machine learning. It automates the creation of an ensemble pipeline containing a selected number of models, optimizing hyperparameters for optimal prediction scores.

## Input Parameters

When initializing the `ModelSelector` class, you can provide the following input parameters:

- `data` (pandas DataFrame): The input dataset to be used for model selection.

- `target` (str): The name of the target column in the dataset.

- `eda_pipe` (sklearn Pipeline, optional): An optional pipeline for Exploratory Data Analysis (EDA). If provided, EDA will be performed on the dataset before model selection. Defaults to `None`.

- `task` (str, optional): The task to be performed, which can be either 'classification' or 'regression'. Defaults to 'classification'.

- `i` (int, optional): The number of best models to be selected. You can specify the number of top-performing models to include in the final ensemble. The maximum value is 6. Be aware that selecting more models increases the computation time. Defaults to 2.

- `precision` (float, optional): A precision value ranging from 0.1 to 1. A higher precision value results in more precise model selection but requires more time to run. It determines the granularity of model evaluation. Defaults to 0.2.

## Getting the Final Pipeline

After initializing the `ModelSelector` class with your desired parameters, you can obtain the final machine learning pipeline by calling the `get_pipeline()` method. This pipeline will consist of the selected ensemble of models, fine-tuned with the best hyperparameters.

Here's an example of how to use the `ModelSelector` class:

```python
# Import the ModelSelector class
from model_selector import ModelSelector

# Initialize the ModelSelector with your data and target column
selector = ModelSelector(data=my_data, target='target_column')

# Get the final machine learning pipeline
final_pipeline = selector.get_pipeline()

## Features

- **Automated Ensemble Creation:** The `ModelSelector` class automatically generates an ensemble pipeline with a specified number of models, each contributing to the final predictions.
- **Hyperparameter Optimization:** Utilizes a combination of model selection and hyperparameter tuning to output the best-performing models and their corresponding hyperparameters.
- **Versatile Usage:** Offers both automatic ensemble creation (`start()`) and the option to fine-tune an existing pipeline with a specific model (`auto_tuning()`).
- **Scoring:** Run the (`evaluate()`) function after ensemble creation to get your scores.
- **Supports Classification and Regression:** Adaptable for both classification and regression tasks, providing flexibility in application.
- **Easy Retrieval of Best Pipeline:** Use the `get_pipeline()` function to retrieve the optimized pipeline with the best-performing models.

## Installation

You can install the `ModelSelector` class using pip:

```bash
pip install yctmodel

