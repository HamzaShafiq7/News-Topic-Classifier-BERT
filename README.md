# 📰 News Topic Classifier Using BERT

A fine-tuned BERT transformer model that classifies news headlines into topic categories with high accuracy. Built as part of the AI/ML Engineering Advanced Internship at DevelopersHub Corporation.

---

## 📌 Objective
Fine-tune the `bert-base-uncased` transformer model on the AG News dataset to automatically classify news headlines into 4 categories: **World, Sports, Business, and Sci/Tech**.

---

## 🗂️ Dataset
* **Name:** AG News Dataset
* **Source:** Hugging Face Datasets (`ag_news`)
* **Size:** 4,000 training samples, 800 test samples (subset used for efficient Colab training)
* **Categories:** World (0), Sports (1), Business (2), Sci/Tech (3)

---

## 🏗️ Methodology / Approach

### 1. Data Preparation
* Loaded AG News dataset directly from Hugging Face
* Tokenized headlines using `BertTokenizer` with max length of 128 tokens
* Applied padding and truncation for uniform input size

### 2. Model Fine-Tuning
* Base model: `bert-base-uncased` (110M parameters)
* Added a classification head for 4 output classes
* Fine-tuned using Hugging Face `Trainer` API
* Training config: 3 epochs, batch size 16, learning rate 2e-5, fp16 mixed precision

### 3. Evaluation
* Measured Accuracy and Weighted F1-Score on the test set
* Generated a full classification report per category
* Visualised results with a confusion matrix heatmap

### 4. Deployment
* Deployed with **Gradio** for live interaction
* Public URL generated automatically via `share=True` — no tunnel setup needed
* Interface shows confidence scores for all 4 categories per headline

---

## 📊 Evaluation Metrics
| Metric | Value |
|---|---|
| Accuracy | 90.12% |
| Weighted F1-Score | 90.11% |

---

## 📈 Key Results & Observations
* BERT achieved strong classification accuracy with only 4,000 training samples — demonstrating the power of transfer learning
* Sci/Tech and Business categories showed the most overlap in misclassifications
* fp16 mixed-precision training halved GPU memory usage on Colab T4
* Gradio deployment provided a clean, interactive interface with zero additional configuration

---

## 🛠️ Tech Stack
* Python, PyTorch
* Hugging Face Transformers & Datasets
* Gradio (deployment)
* scikit-learn (evaluation metrics)
* Google Colab T4 GPU

---

## ⚙️ Setup & Run

### Requirements
```bash
pip install transformers datasets scikit-learn accelerate gradio
```

### Environment
> Run on **Google Colab with T4 GPU**
> `Runtime → Change runtime type → T4 GPU`

### Steps
1. Open `News_Topic_Classifier_BERT.ipynb` in Google Colab
2. Run all cells in order — training takes ~10–15 minutes on T4
3. The Gradio deployment cell will generate a public link automatically

---

## 📁 Repository Structure
```
├── BERT_News_Classifier.ipynb
└── README.md
```
