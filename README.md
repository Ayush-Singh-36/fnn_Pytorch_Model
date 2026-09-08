## Project Overview

This project implements a deep learning model to predict software developer salaries based on various features such as experience, country, education, programming languages, frameworks used, and company size. The model leverages PyTorch for building a custom neural network that handles both categorical and numerical input features effectively.

## Features

- **Data Loading & Preprocessing**: Handles dataset loading from Kaggle, includes data profiling for understanding feature cardinality, and preprocesses data (scaling numerical features, mapping categorical features to integers).
- **Custom PyTorch Model**: Implements a `SalaryPredictionModel` using `nn.Embedding` layers for categorical features and a fully connected neural network (MLP) for regression.
- **Embedding Layers**: Utilizes embedding layers to convert high-cardinality categorical features into dense, lower-dimensional vectors, improving model performance and generalization.
- **Batch Normalization & Dropout**: Incorporates `BatchNorm1d` and `Dropout` layers to prevent overfitting and stabilize training.
- **Training Loop**: Provides a clear training loop with `MSELoss` and `Adam` optimizer, tracking training loss and RMSE.
- **Model Export**: Includes functionality to export the trained model to ONNX format for potential deployment (Note: ONNX export currently has a dependency issue).

## Technologies Used

- Python 3.x
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Kaggle API (for dataset download)
- ONNX (for model export)

## Setup

1.  **Clone the repository (or open in Colab):**
    ```bash
    git clone https://github.com/Ayush-Singh-36/fnn_Pytorch_Model
    cd fnn_Pytorch_Model
    ```
2.  **Kaggle API Key**: This notebook downloads the dataset directly from Kaggle. You need to set up your Kaggle API credentials:
    - Go to your Kaggle account settings and create a new API token (this will download `kaggle.json`).
    - In Google Colab, go to `Secrets` (the key icon on the left panel) and add two new secrets:
        - `KAGGLE_USERNAME` (your Kaggle username)
        - `KAGGLE_KEY` (the API key from `kaggle.json`)

## Usage

1.  **Run all cells in the Jupyter/Colab notebook.**
2.  The notebook will:
    - Set up a device-agnostic PyTorch environment (CUDA if available, else CPU).
    - Download the dataset from Kaggle.
    - Profile `data_dictionary.csv`, `train.csv`, and `test.csv` to understand their structure and cardinality.
    - Create `MyDataset` and `DataLoader` instances for training and testing.
    - Define and initialize the `SalaryPredictionModel`.
    - Train the model for 100 epochs, printing training loss and RMSE periodically.
    - Attempt to export the trained model to ONNX format.

## Model Details

The `SalaryPredictionModel` is a custom neural network that takes two types of inputs:

-   **Categorical Inputs (`x_cat`)**: Processed through `nn.Embedding` layers. Each categorical feature (`country`, `education`, `languages`, `frameworks`, `company_size`) has its own embedding layer. The embedding dimensions are determined dynamically based on the number of unique categories.
-   **Numerical Inputs (`x_num`)**: Currently includes `experience_scaled`.

The embedded categorical features are concatenated with the numerical features and then passed through a sequential fully connected network (`fc_net`) consisting of:

-   `nn.Linear` layers
-   `nn.BatchNorm1d` layers for normalization
-   `nn.ReLU` activation functions
-   `nn.Dropout` layers for regularization

The final layer outputs a single value representing the predicted salary.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests for any improvements or bug fixes.

## License

MIT License
