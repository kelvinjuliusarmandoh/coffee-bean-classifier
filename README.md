<h1>Coffee Bean Classifier</h1>


<h2>Overview</h2>
<p>The Coffee Bean Classifier is a machine learning project designed to classify different types of coffee beans based on various features such as color, size, and shape. 
  The goal is to help coffee producers, roasters, and distributors quickly identify the quality and type of coffee beans to ensure consistent product quality.</p>

<p>There are four roasting levels. The green or un-roasted coffee beans are Laos Typica Bolaven (Coffea arabica). 
  Laos Typica Bolaven is a lightly roasted coffee bean (Coffea arabica). 
  Doi Chaang (Coffea Arabica) is medium roasted, whereas Brazil Cerrado is dark roasted (Coffea Arabica).</p>

<h2>Key Features</h2>
* Data Preprocessing
  Includes Resize the image into 224x224 pixels and transform to floating number around [0, 1]
* Deep Learning Model: Employs a CNN (Convolutional Neural Network) for high-performance image classification.
* Evaluation Metrics: Comprehensive evaluation using accuracy.

<h2>Prerequisites</h2>
* Python 3.10.13
* PyTorch
* Streamlit
* Matplotlib

<h2>Project structure</h2>
  ```
  coffee-bean-classifier/
  │
  ├── data/                 # Directory for dataset
  ├── models/               # Saved models
  ├── notebooks/            # Jupyter notebooks for exploratory data analysis
  ├── Modular/                  # Source code for the project
  │   ├── data_setup.py
  │   ├── engine.py
  │   ├── model_builder.py
  │   └── train.py
  ├── results/              # Directory to save model results and logs
  ├── README.md             # Project README file
  └── requirements.txt      # Python dependencies
  ```
  
<h2>Model Description</h2>
The Coffee Bean Classifier is based on a Convolutional Neural Network (CNN) architecture optimized for image classification tasks. The model was trained on a dataset of labeled coffee bean images, with the following layers:
* Input Layer: Processes the input image (e.g., 128x128 RGB).
* Convolutional Layers: Extracts features from the images.
* Pooling Layers: Reduces the dimensionality while retaining important features.
* Fully Connected Layers: Classifies the extracted features into different coffee bean types.
* Output Layer: Provides the final classification result with probabilities.

Results:
![image](https://github.com/user-attachments/assets/66f4afc2-cc5f-4ac8-a2f2-640be1b9d0f4)
