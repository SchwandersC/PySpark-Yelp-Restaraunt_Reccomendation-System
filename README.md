# Yelp Rating Prediction ML Project

This project leverages user and business data from Yelp to predict review ratings. It combines advanced feature engineering—including selecting the top 15 features most correlated with the target—with an XGBoost regression model to improve prediction accuracy.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Feature Selection](#feature-selection)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The main goal of this project is to predict Yelp review ratings by integrating features extracted from `user.json` and `business.json`. The pipeline includes:
- **Data Loading:** Reading JSON and CSV files using PySpark.
- **Feature Engineering:** Extracting and flattening features, handling missing values, and encoding categorical variables.
- **Feature Selection:** Computing Pearson correlations between each feature and the target rating, then selecting the top 15 features based on absolute correlation values.
- **Model Training:** Training an XGBoost regressor on the selected features.
- **Prediction:** Generating predictions on a validation set and outputting the results to a CSV file.

## Project Structure

```plaintext
ml_project/
├── data/                       # Data files (JSON and CSV)
│   ├── business.json          # Business data
│   ├── user.json              # User data
│   ├── yelp_train.csv         # Training review data
│   └── yelp_val_in.csv        # Validation review data
├── src/                        # Source code for the project
│   ├── __init__.py             
│   ├── config.py              # Configuration settings and hyperparameters
│   ├── data_loader.py         # Data loading functions using PySpark
│   ├── feature_engineering.py # Feature extraction, encoding, and selection
│   └── model.py               # Model training and prediction functions
├── requirements.txt           # List of dependencies
├── run.py                     # Main entry point of the project
└── README.md                  # This file
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd ml_project
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the project by providing the paths to your training data directory, validation CSV file, and output file for predictions. For example:

```bash
python run.py ./data/ ./data/yelp_val_in.csv ./output_predictions.csv
```

### Command Line Arguments
- `<train_data_path>`: Directory containing the training files (`business.json`, `user.json`, `yelp_train.csv`).
- `<val_data_path>`: Path to the validation CSV file (e.g., `yelp_val_in.csv`).
- `<output_file>`: Path to save the predictions CSV file.

## Configuration

Model hyperparameters and configuration settings are defined in the [`src/config.py`](src/config.py) file. You can tweak parameters such as:
- `max_depth`
- `learning_rate`
- `n_estimators`
- etc.

to experiment with different model setups.

## Feature Selection

In the feature engineering stage, after encoding the data:
- The Pearson correlation between each feature and the target rating is computed.
- Features are sorted by the absolute correlation value.
- The top 15 features are selected and used to train the model.
  
This step helps in reducing noise and focusing the model on the most predictive features.

## Dependencies

- **Python 3.x**
- **PySpark**
- **XGBoost**
- **NumPy**
- **scikit-learn**

Refer to the [`requirements.txt`](requirements.txt) file for the full list of dependencies.

## Results

Error Distribution: 

>=0 and <1: 102264
>=1 and <2: 32858
>=2 and <3: 6109
>=3 and <4: 811
>=4: 2

RMSE: 0.978170924312778

Execution Time: 175 sec

## License

This project is licensed under the [MIT License](LICENSE).

