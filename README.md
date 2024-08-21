# Coffee Bean Classifier


## Overview
The Coffee Bean Classifier is a machine learning project designed to classify different types of coffee beans based on various features such as color, size, and shape. 
The goal is to help coffee producers, roasters, and distributors quickly identify the quality and type of coffee beans to ensure consistent product quality.

There are four roasting levels:
- The green or un-roasted coffee beans are Laos Typica Bolaven (Coffea arabica). 
- Laos Typica Bolaven is a lightly roasted coffee bean (Coffea arabica). 
- Doi Chaang (Coffea Arabica) is medium roasted.
- Brazil Cerrado is dark roasted (Coffea Arabica).

## Key Features 
- Data Preprocessing
  Includes Resize the image into 224x224 pixels and transform to floating number around [0, 1]
- Deep Learning Model: Employs a CNN (Convolutional Neural Network) for high-performance image classification.
- Evaluation Metrics: Comprehensive evaluation using accuracy.

## Prerequisites
- Python 3.10.13
- PyTorch
- Streamlit
- Matplotlib

## Project structure 
```
coffee-bean-classifier/
│
├── data/                 # Directory for dataset
├── models/               # Saved models
├── notebooks/            # Jupyter notebooks for exploratory data analysis
├── Modular/              # Source code for the project
│   ├── data_setup.py     # Script for setting up data loaders
│   ├── engine.py         # Script for training engine
│   ├── model_builder.py  # Script for building the model
│   └── train.py          # Script for training the model
├── results/              # Directory to save model results and logs
├── README.md             # Project README file
└── requirements.txt      # Python dependencies
```

## Model Description 
The Coffee Bean Classifier is based on a Convolutional Neural Network (CNN) architecture optimized for image classification tasks. The model was trained on a dataset of labeled coffee bean images, with the following layers:
- Input Layer: Processes the input image (e.g., 128x128 RGB).
- Convolutional Layers: Extracts features from the images.
- Pooling Layers: Reduces the dimensionality while retaining important features.
- Fully Connected Layers: Classifies the extracted features into different coffee bean types.
- Output Layer: Provides the final classification result with probabilities.

## Results:

![image](https://github.com/user-attachments/assets/66f4afc2-cc5f-4ac8-a2f2-640be1b9d0f4)
