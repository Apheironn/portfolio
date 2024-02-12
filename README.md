---
permalink: /portfolio/
title: "Portfolio"
toc: true
toc_label: "Table of Contents"
toc_icon: "bookmark"

---
*Updated: 12/02/2024*

## 🤖 Computer Vision
### Object Detection with YOLO

<img src="images/moon.png" width="200">

A series of notebooks to dive deep into popular datasets for object detection and learn how to train Detectron2 on custom datasets.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/object-detection-detectron2)

- [Notebook 4](https://github.com/chriskhanhtran/object-detection-detectron2/blob/master/04-train-big.ipynb): Train Detectron2 on Open Images dataset to detect musical instruments.
- [Notebook 5](https://github.com/chriskhanhtran/object-detection-detectron2/blob/master/05-kaggle-global-wheat-detection.ipynb): Apply Detectron2 on [Kaggle Global Wheat Detection Competition](https://www.kaggle.com/c/global-wheat-detection).

<img src="https://raw.githubusercontent.com/chriskhanhtran/object-detection-detectron2/master/images/output_04.png" width="580">{: .align-center}

### Visual Recognition for Vietnamese Foods

[![Open Web App](https://img.shields.io/badge/Heroku-Open_Web_App-blue?logo=Heroku)](https://vietnamese-food.herokuapp.com/)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://chriskhanhtran.github.io/posts/vn-food-classifier/)

I built a Computer Vision application to recognize popular Vietnamese dishes and display their information and stories. To build the application, I crawled 6,000 images using the Bing Image Search API and trained a ResNet-50 model. The model achieved 94% accuracy over 11 classes.

<img src="https://github.com/chriskhanhtran/vn-food-app/blob/master/img/vn-food-app.gif?raw=true" width="580">{: .align-center}

### CS231n: Convolutional Neural Networks for Visual Recognition

This is my complete implementation of assignments and projects in [***CS231n: Convolutional Neural Networks for Visual Recognition***](http://cs231n.stanford.edu/) by Stanford (Spring, 2019). NumPy implementations of forward and backward pass of each layer in a convolutional neural network have given me a deep understanding of how state-of-the-art Computer Vision architectures work under the hood. In addition, I explored the inner beauty of Deep Learning by implementing Style Transfer, Deep Dream, Texture Systhesis in PyTorch and generating new images with GANs.

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/CS231n-CV)

**Selective Topics:**
- [NumPy Implementations of CNN](https://github.com/chriskhanhtran/CS231n-CV/blob/master/assignment2/cs231n/layers.py): Fully-connected Layer, Batchnorm, Layernorm, Dropout, Convolution, Maxpool.
- [Image Captioning with LSTMs](https://github.com/chriskhanhtran/CS231n-CV/blob/master/assignment3/LSTM_Captioning.ipynb)
- [Saliency Maps, Deep Dream, Fooling Images](https://github.com/chriskhanhtran/CS231n-CV/blob/master/assignment3/NetworkVisualization-PyTorch.ipynb)
- [Style Transfer](https://github.com/chriskhanhtran/CS231n-CV/blob/master/assignment3/StyleTransfer-PyTorch.ipynb)
- [Generative Adversarial Networks (GANs)](https://github.com/chriskhanhtran/CS231n-CV/blob/master/assignment3/Generative_Adversarial_Networks_PyTorch.ipynb)


## 📈 Data Science
### Credit Risk Prediction Web App

[![Open Web App](https://img.shields.io/badge/Heroku-Open_Web_App-blue?logo=Heroku)](http://credit-risk.herokuapp.com/)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/chriskhanhtran/credit-risk-prediction/blob/master/documents/Notebook.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/credit-risk-prediction)

After my team preprocessed a dataset of 10K credit applications and built machine learning models to predict credit default risk, I built an interactive user interface with Streamlit and hosted the web app on Heroku server.

<img src="https://chriskhanhtran.github.io/minimal-portfolio/images/credit-risk-webapp.png" width="580">{: .align-center}

---
### Kaggle Competition: Predict Ames House Price using Lasso, Ridge, XGBoost and LightGBM

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://chriskhanhtran.github.io/minimal-portfolio/projects/ames-house-price.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/kaggle-house-price/blob/master/ames-house-price.ipynb)

I performed comprehensive EDA to understand important variables, handled missing values, outliers, performed feature engineering, and ensembled machine learning models to predict house prices. My best model had Mean Absolute Error (MAE) of 12293.919, ranking **95/15502**, approximately **top 0.6%** in the Kaggle leaderboard.

<img src="https://chriskhanhtran.github.io/assets/images/portfolio/ames-house-price.jpg" width="580">{: .align-center}

---
### Predict Breast Cancer with RF, PCA and SVM using Python

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://chriskhanhtran.github.io/minimal-portfolio/projects/breast-cancer.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/predict-breast-cancer-with-rf-pca-svm/blob/master/breast-cancer.ipynb)

In this project I am going to perform comprehensive EDA on the breast cancer dataset, then transform the data using Principal Components Analysis (PCA) and use Support Vector Machine (SVM) model to predict whether a patient has breast cancer.

<img src="https://chriskhanhtran.github.io/minimal-portfolio/images/breast-cancer.png" width="580">{: .align-center}

---
### Business Analytics Conference 2018: How is NYC's Government Using Money?

[![Open Research Poster](https://img.shields.io/badge/PDF-Open_Research_Poster-blue?logo=adobe-acrobat-reader&logoColor=white)](https://chriskhanhtran.github.io/minimal-portfolio/pdf/bac2018.pdf)

In three-month research and a two-day hackathon, I led a team of four students to discover insights from 6 million records of NYC and Boston government spending data sets and won runner-up prize for the best research poster out of 18 participating colleges.

<img src="https://chriskhanhtran.github.io/assets/images/portfolio/bac2018.JPG" width="580">{: .align-center}
