# 🔮 DSLR (Data Science Logistic Regression)

![Language](https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white)
![Library](https://img.shields.io/badge/Library-Pandas%20%7C%20Numpy%20%7C%20Matplotlib-orange)

## Overview

**DSLR** is a Data Science and Machine Learning project that implements Logistic Regression from scratch to solve a classification problem: sorting Hogwarts students into their respective houses (Gryffindor, Hufflepuff, Ravenclaw, Slytherin) based on their academic scores.

This project mimics the functionality of the "Sorting Hat" using machine learning algorithms, specifically **One-vs-All** logistic regression trained via **Gradient Descent**.

## 📊 Features

-   **Data Analysis**: Custom implementation of descriptive statistics (mean, std, min, max, quartiles).
-   **Visualization**:
    -   **Histograms**: To analyze score distribution across houses.
    -   **Scatter Plots**: To observe feature correlations.
    -   **Pair Plots**: To visualize the entire dataset structure.
-   **Machine Learning**:
    -   **Logistic Regression**: One-vs-All classification.
    -   **Optimization Algorithms**:
        -   **Batch Gradient Descent**: Stable updates using the entire dataset.
        -   **Stochastic Gradient Descent (SGD)**: Faster updates using single examples.
        -   **Mini-Batch Gradient Descent**: Balanced approach.
    -   **Loss Function**: **Cross-Entropy Loss** (Log Loss) to minimize error.
    -   **Accuracy**: High precision classification (>98%).

## 🛠️ Installation

Ensure you have Python 3 installed along with the required libraries:

```bash
pip install pandas numpy matplotlib seaborn
```

## 🚀 Usage

### 1. Data Exploration

Analyze the dataset using the provided tools:

```bash
# Display descriptive statistics (similar to pandas .describe())
python3 describe.py datasets/dataset_train.csv

# Generate histograms for feature analysis
python3 histogram.py datasets/dataset_train.csv

# Generate scatter plots to find correlated features
python3 scatter.py datasets/dataset_train.csv

# Generate a pair plot for global overview
python3 pair_plot.py datasets/dataset_train.csv
```

### 2. Training

Train the model using the training dataset. You must specify the gradient descent method (`batch`, `stochastic`, or `mini_batch`).

```bash
# Usage: python3 logreg_train.py <dataset> <mode>
python3 logreg_train.py datasets/dataset_train.csv mini_batch
```

### 3. Prediction

Use the trained weights to predict the houses for the test dataset. This generates a `houses.csv` file.

```bash
python3 logreg_predict.py datasets/dataset_test.csv weights.pkl
```

### 4. Evaluation

Compare the predictions with the actual results (if available) to calculate accuracy.

```bash
python3 compare.py houses.csv datasets/dataset_truth.csv
```

## 📂 Project Structure

-   `describe.py`: Statistical analysis tool.
-   `histogram.py`, `scatter.py`, `pair_plot.py`: Visualization tools.
-   `logreg_train.py`: Training algorithm (Gradient Descent).
-   `logreg_predict.py`: Prediction script.
-   `datasets/`: Contains training and testing CSV files.
