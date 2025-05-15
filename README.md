# DetoxAI: Detection and Mitigation of Biases in Language Models and Datasets

**DetoxAI** is a system designed to identify and mitigate biases in language models and datasets using an integrated framework that includes GAN-based discrimination, fairness toolkits, and debiasing pipelines. The system evaluates gender, age, and social bias in both datasets and model predictions, and provides mechanisms for mitigation.

---

## Motivation

Language models often inherit or amplify biases present in their training data. These biases can lead to unfair outcomes in real-world applications. DetoxAI is developed to:

- Detect biases in datasets and pretrained models.
- Apply GAN-based discrimination for bias quantification.
- Support dataset and model debiasing pipelines.
- Empower users to evaluate their models and data before deployment.

---

## Features

- **Bias Detection**: Detects gender, age, and social biases using a GAN-based architecture.
- **Dataset Debiasing**: Uses AIF360 and FairLens to identify and debias datasets.
- **Model Evaluation**: Works with HuggingFace-based models (BERT, RoBERTa, T5, etc.).
- **GAN Discriminator Module**: Compares real model predictions with synthetic unbiased outputs.
- **Image-to-Text Bias Detection**: Via OpenBias integration for VQA tasks.
- **Bias Score Calculation**: Quantitative metric using discriminator's confidence difference.
- **Extensible Architecture**: Can be adapted to support more model types or modalities.


