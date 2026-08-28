# Deep Learning Based Detection of Fake News on Sri Lanka in the Sinhala Language with Automated Legal Penalty Notification


## 📌 Project Overview

This research project focuses on developing a Deep Learning-based system for detecting fake news written in the Sinhala language within the Sri Lankan context.

The proposed system classifies Sinhala news content as **Real or Fake** using Natural Language Processing (NLP) and Deep Learning techniques. The system also includes an automated legal penalty notification component that provides relevant legal information when content is classified as fake.

---

## 🎯 Research Aim

To develop and evaluate a Deep Learning-based system for detecting Sinhala fake news related to Sri Lanka and provide automated legal penalty information for detected fake news.

---

## 🔍 Research Objectives

1. To collect and prepare a Sinhala fake news dataset containing real and fake news samples.
2. To preprocess Sinhala text using appropriate Natural Language Processing techniques.
3. To investigate different text representation methods for Sinhala news classification.
4. To develop Deep Learning models for Sinhala fake news detection.
5. To compare the performance of different embedding and Deep Learning model combinations.
6. To evaluate the developed models using standard classification metrics.
7. To develop a prototype system with an automated legal penalty notification mechanism.

---

## 🗂️ Dataset

The research aims to develop a dataset containing approximately **1,500 Sinhala news samples**, including both real and fake news.

### Classes

- Real News
- Fake News

### Data Sources

Real news data are collected from reliable Sinhala news sources and official sources, while fake news data are collected from publicly available social media content and fact-checking resources.

---

## 🧹 Data Pre-processing

The collected Sinhala text is preprocessed before being provided to the Deep Learning models.

The main preprocessing steps include:

- Text Cleaning
- URL and unwanted character removal
- Tokenization
- Stop-word Removal
- Unicode Normalization
- Dataset Splitting

### Dataset Split

- 70% Training
- 15% Validation
- 15% Testing

---

## 🧠 Text Representation / Embeddings

Three text representation techniques are investigated:

### 1. FastText

FastText represents words using sub-word information and is useful for languages with morphological variations and limited linguistic resources.

### 2. mBERT

Multilingual BERT (mBERT) is a multilingual Transformer-based model that provides contextual representations of text.

### 3. XLM-R

XLM-R (XLM-RoBERTa) is a multilingual Transformer-based language model designed for cross-lingual Natural Language Processing tasks.

---

## 🤖 Deep Learning Models

The following Deep Learning architectures are investigated:

- TextCNN
- LSTM
- BiLSTM
- GRU

### Model Combinations

3 Embedding Methods × 4 Deep Learning Models = **12 Combinations**

The combinations include:

- FastText + TextCNN
- FastText + LSTM
- FastText + BiLSTM
- FastText + GRU
- mBERT + TextCNN
- mBERT + LSTM
- mBERT + BiLSTM
- mBERT + GRU
- XLM-R + TextCNN
- XLM-R + LSTM
- XLM-R + BiLSTM
- XLM-R + GRU

---

## ⚙️ Technologies Used

### Programming Language

- Python

### Deep Learning

- PyTorch

### NLP / Transformers

- Hugging Face Transformers
- Gensim
- FastText
- mBERT
- XLM-R

### Data Processing

- Pandas
- NumPy
- Scikit-learn

### Data Collection

- BeautifulSoup
- Scrapy

### Development Environment

- Google Colab
- Jupyter Notebook

---

## 📊 Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Macro F1-Score
- Confusion Matrix
- ROC-AUC

The Macro F1-Score is considered an important metric for comparing model performance because the dataset may contain class imbalance.

---

## ⚖️ Automated Legal Penalty Notification

When a submitted news item is classified as **Fake**, the system provides relevant legal penalty information.

### System Flow

User submits Sinhala News  
↓  
Text Pre-processing  
↓  
Text Embedding  
↓  
Deep Learning Model  
↓  
Real / Fake Classification  
↓  
If Fake → Display Legal Penalty Information

The legal notification component is intended for informational purposes only and does not constitute legal advice or a formal legal determination.

---

## 🏗️ Proposed System Architecture

```text
              Sinhala News Input
                      |
                      v
              Data Pre-processing
                      |
                      v
              Text Representation
           +----------+----------+
           |          |          |
           v          v          v
        FastText     mBERT      XLM-R
           |          |          |
           +----------+----------+
                      |
                      v
             Deep Learning Models
        +---------+---------+---------+
        |         |         |         |
        v         v         v         v
     TextCNN    LSTM     BiLSTM      GRU
        |         |         |         |
        +---------+---------+---------+
                      |
                      v
                REAL / FAKE
                      |
                      v
          Legal Penalty Notification
