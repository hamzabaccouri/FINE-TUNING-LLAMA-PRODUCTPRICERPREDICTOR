
# Fine-Tuning LLaMA Product Pricer Predictor

This repository contains the implementation for fine-tuning the **LLaMA** language model to predict product prices based on their descriptions. The project leverages **QLoRA (Quantized LoRA)** for efficient fine-tuning with limited resources, including 4-bit quantization.

## Repository Structure

```
FINE-TUNING-LLAMA-PRODUCTPRICERPREDICTOR/
├── helpers/
│   ├── items.py         # Core logic for handling data and items
│   ├── loaders.py       # Utility functions for data loading
├── notebooks/
│   ├── LLAMAProductPricer.ipynb                 # Core model implementation
│   ├── PrepareTrainingData.ipynb               # Notebook for preprocessing training data
│   ├── QLORA_FineTuning_LLAMAProductPricer.ipynb # QLoRA fine-tuning process
│   ├── testing_Finetuned_LLAMAProductPricer.ipynb # Testing the fine-tuned model
├── .env                # Environment variables for sensitive information (e.g., API keys)
├── requirements.txt    # Python dependencies for the project
├── train_lite.pkl      # Training dataset (lite version)
├── test_lite.pkl       # Testing dataset (lite version)
```

## Features

- **Efficient Fine-Tuning**: Implements **QLoRA** for low-resource environments with 4-bit quantization.
- **Data Handling**: Scripts and helpers for loading and preprocessing datasets.
- **Price Prediction**: Fine-tunes the **LLaMA** model to predict product prices from their descriptions.
- **Notebooks for Easy Experimentation**: Separate notebooks for preprocessing, fine-tuning, and evaluation.

## Getting Started

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/hamzabaccouri/FINE-TUNING-LLAMA-PRODUCTPRICERPREDICTOR.git
   cd FINE-TUNING-LLAMA-PRODUCTPRICERPREDICTOR
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   - Create a `.env` file in the root directory with the following format:

     ```
     HF_TOKEN=your_hugging_face_token
     ```

   - Replace `your_hugging_face_token` with your Hugging Face token.

### Run Notebooks

1. Start by preparing the training data:

   - Open `notebooks/PrepareTrainingData.ipynb` and run the cells.

2. Fine-tune the LLaMA model:

   - Open `notebooks/QLORA_FineTuning_LLAMAProductPricer.ipynb` and execute the fine-tuning process.

3. Test the fine-tuned model:

   - Open `notebooks/testing_Finetuned_LLAMAProductPricer.ipynb` to evaluate the model on test data.

## Datasets

- The project uses a dataset of Amazon products from Hugging Face, focusing on categories such as:
  - **Software**
  - **Video Games**
  - **Health and Personal Care**
  - **Handmade Products**
  - **Appliances**
- Lite versions of the training and testing datasets (`train_lite.pkl` and `test_lite.pkl`) are included for quick experimentation.
