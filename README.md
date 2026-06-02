# ATP Tennis Match Prediction - Deep Learning Final Project

This repository contains the final project for the Coursera Deep Learning course. The objective is to predict the winner of professional ATP tennis matches using a Deep Neural Network.

## Project Structure
- `data/`: Local storage for raw and preprocessed data (excluded from Git).
- `src/`: Clean, modular Python scripts for processing and training.
- `reports/`: Documentation and final analysis reports.
- `results/`: Performance visualizations and comparison plots.

## Data Acquisition
The raw dataset is not included in this repository to keep it lightweight. To run this project, you need to clone the ATP tennis dataset into the `data/raw` folder:

```bash
mkdir -p data/raw
git clone https://github.com/JeffSackmann/tennis_atp.git data/raw/tennis_atp --depth 1
```

## Setup Instructions
1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Preprocess Data**:
   ```bash
   python3 src/preprocess.py
   ```
2. **Train Models**:
   ```bash
   python3 src/train.py
   ```

## Key Findings
The project evaluates three Multi-Layer Perceptron (MLP) architectures. The model with **Dropout (Model 2)** achieved the best performance with a test accuracy of ~63.4%, demonstrating better generalization than deeper or batch-normalized variations.
