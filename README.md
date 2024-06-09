# Credit Scoring ML Model

This repository contains the code and resources for building a credit scoring model using data provided by an eCommerce platform. The model categorizes users as high risk or low risk, selects observable features as predictors of default, assigns risk probability, assigns credit scores, and predicts the optimal amount and duration of loans.

### Directories
- **notebooks**: Jupyter notebooks for exploratory data analysis and model development.
- **scripts**: Scripts for data preprocessing, feature engineering, and model training.
- **src**: Source code for the core functionality of the credit scoring model.
- **tests**: Unit tests to ensure the correctness and reliability of the code.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/amlaksil/credit-scoring-ml-model.git
		```

2. Navigate to the project directory:
		```bash
		cd credit-scoring-ml-model
		```

3. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

5. Download the dataset from Kaggle and place it in the `data/` directory.

## Usage

To run the main script, which loads data, processes it, calculates RFMS scores, defines features and targets, trains models, and evaluates model performance, use the following command:

```bash
python3 -m scripts.run_model
```

## Results

The performance of the models is summarized below:

| Metric    | Logistic Regression | Random Forest | Gradient Boosting Machines |
|-----------|---------------------|---------------|----------------------------|
| Accuracy  | 88.26%              | 100%          | 100%                       |
| Precision | 85.71%              | 100%          | 100%                       |
| Recall    | 84.11%              | 100%          | 100%                       |
| F1 Score  | 84.91%              | 100%          | 100%                       |
| ROC-AUC   | 95.65%              | 100%          | 100%                       |

## Contributing

Contributions are welcome! If you have any suggestions or changes you would like to discuss, please create an issue first.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
