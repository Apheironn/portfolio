# Table of Contents
- [Computer Vision](#-computer-vision)
  - [Galaxy Images Classification](galaxy-images-classification---tensorflow--flask--docker--aws)
- [Data Science](#-data-science)


*Updated: 13/02/2024*

 
# 🤖 Computer Vision 

### Galaxy Images Classification - Tensorflow | Flask | Docker | AWS
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Apheironn/End-to-End-Galaxy-Classification)

This project focuses on using deep learning techniques to classify different types of galaxies based on their images. The goal is to develop a machine learning model that can accurately distinguish between various galaxy shapes and configurations.

- Developing a galaxy classification model that recognizes various galaxy types with 90% accuracy
- Creating a user-friendly interface to increase astronomical understanding
- Ensure scalability and consistencywith Docker for efficient deployment
- Provide global availability by distributing the project on AWS EC2 for broad use

<img src="images/Pipeline.png" width="700">

### Automated Component Segmentation in Production Lines

- Implemented a robust computer vision solution utilizing OpenCV to identify and segment 15 distinct components on a production line, ensuring high-precision part recognition and sorting.
- Developed an algorithm using the ORB (Oriented FAST and Rotated BRIEF) feature detector to create descriptors for each component, coupled with a BFMatcher for accurate matching, allowing for real-time component identification.
- Successfully deployed the system to operate directly on the production line, demonstrating high accuracy in component separation, significantly optimizing the manufacturing process and reducing manual inspection workload.

![Object Detection](video/opencv.gif)

### Korug - Robotic Drone
Korug is a robotic drone that has dual air and land maneuverability and can perform cyber attacks and enter-and-take out operations; It came first in the first-class engineering project competition by presenting applications in the fields of
military, agriculture, disaster relief and health.

![Drive via PS4](video/surus.gif)

### Moon Shape Classification using YOLO and Roboflow

Developed a computer vision model using YOLO and Roboflow for moon shape classification.

- Handled a dataset of 1370 images with 8 classes.
- Implemented preprocessing techniques like auto-orient and resizing to 640x640 pixels.
- Enhanced dataset quality with augmentations: grayscale, saturation, brightness, exposure adjustments, blur, and noise.
- Calculated an F1 Score of approximately 81.0%, balancing precision and recall, signifying a robust model performance in moon shape classification.

<img src="images/moon.png" width="500">

### Object Detection App

The object detection app serves as a practical implementation of the MobileNet AI model, a cutting-edge neural network architecture that brings efficiency to on-device processing. Developed with a quantization scheme tailored for enhanced on-device performance, the model has undergone rigorous testing on MobileNets. Its prowess is evident in the notable improvements it brings to ImageNet classification and COCO detection on widely used CPUs.

<img src="images/mobile1.png">

<img src="images/mobile.png" width="500">



# 📈 Data Science
### End-to-End Car Price Regression - MLflow | Flask
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Apheironn/End-to-End-Machine-Learning-MLflow-Project)

- Developed a regression model to predict car prices using 19 columns as features.
- Utilized MLflow to track experiments, including parameter tuning, model performance, and version control.
- Achieved an R-squared value of %79.
- Implemented DecisionTreeRegressor to establish relationships between features and car price.

<img src="images/gui.png" width="700">
<img src="images/mlflow.png" width="700">

### Time Series Analysis for Markets

- The dataset includes information on products sold in different stores over a specific time period from January 1, 2015, to July 31, 2015.
- The primary objective of this analysis is to understand how promotions and other factors might influence sales in different stores and for various products. By studying the data, we aim to provide insights and recommendations that can help optimize promotion strategies and improve overall sales performance.
<img src="images/invent.png" width="700">


### End-to-End House Price Regression - Docker | Flask

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Apheironn/End-to-End-Machine-Learning-House-Price-Predict-Project)

- Engineered a predictive model to estimate house prices based on 12 distinct variables.
- Employed CatBoost, a gradient boosting framework, to train the regression model owing to its high performance and handling of categorical features.
- Packaged the application using Docker to ensure consistent environments and reproducibility across different systems.

<img src="images//webapp.png" width="700">
