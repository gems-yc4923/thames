# ModelSelector

The `ModelSelector` class is a comprehensive tool designed to simplify the model selection process in machine learning. It automates the creation of an ensemble pipeline containing a selected number of models, optimizing hyperparameters for optimal prediction scores.

## Features

- **Automated Ensemble Creation:** The `ModelSelector` class automatically generates an ensemble pipeline with a specified number of models, each contributing to the final predictions.
- **Hyperparameter Optimization:** Utilizes a combination of model selection and hyperparameter tuning to output the best-performing models and their corresponding hyperparameters.
- **Versatile Usage:** Offers both automatic ensemble creation (`start()`) and the option to fine-tune an existing pipeline with a specific model (`auto_tuning()`).
- **Supports Classification and Regression:** Adaptable for both classification and regression tasks, providing flexibility in application.
- **Easy Retrieval of Best Pipeline:** Use the `get_pipeline()` function to retrieve the optimized pipeline with the best-performing models.

## Installation

You can install the `ModelSelector` class using pip:

```bash
pip install yctmodel

