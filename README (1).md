# Industrial Surface Defect Detection using Deep Learning

An AI-powered computer vision system for automatic detection and
classification of industrial surface defects using Convolutional Neural
Networks (CNNs) and Transfer Learning.

------------------------------------------------------------------------

## 📌 Project Overview

Manual inspection of industrial products is time-consuming, expensive,
and prone to human error. This project automates the quality inspection
process by detecting defects on industrial surfaces using deep learning
techniques.

The model is trained on an industrial defect dataset and classifies
images into different defect categories with high accuracy. The system
can be integrated into manufacturing production lines for real-time
quality control.

------------------------------------------------------------------------

## 🎯 Objectives

-   Automate industrial surface inspection
-   Reduce human inspection errors
-   Improve manufacturing quality control
-   Detect multiple surface defects accurately
-   Build a scalable AI-based inspection system

------------------------------------------------------------------------

## 🛠️ Technologies Used

-   Python
-   TensorFlow
-   Keras
-   OpenCV
-   NumPy
-   Matplotlib
-   Scikit-learn
-   Google Colab / Kaggle

------------------------------------------------------------------------

## 📂 Dataset

The project uses an Industrial Surface Defect Dataset containing images
of various manufacturing defects.

### Dataset Structure

``` text
dataset/
│
├── Class_1
├── Class_2
├── Class_3
├── ...
└── Class_N
```

Each folder contains images belonging to one defect category.

------------------------------------------------------------------------

## 🔄 Data Preprocessing

The following preprocessing pipeline is applied before training:

-   Resize all images to **256 × 256**
-   Convert grayscale images into **3-channel RGB**
-   Normalize pixel values
-   Apply data augmentation
    -   Rotation
    -   Zoom
    -   Horizontal Flip
    -   Width Shift
    -   Height Shift
-   Split dataset into Training and Validation sets

------------------------------------------------------------------------

## 🧠 Model Architecture

The project utilizes **Transfer Learning** with pretrained CNN models.

``` text
Input Image (256×256×3)

        │

Pretrained CNN Backbone

        │

Global Average Pooling

        │

Dropout

        │

Dense Layer (ReLU)

        │

Softmax Output Layer
```

------------------------------------------------------------------------

## 📈 Training Process

-   Loss Function: Categorical Crossentropy
-   Optimizer: Adam
-   Evaluation Metric: Accuracy
-   Early Stopping
-   Model Checkpoint
-   Learning Rate Scheduler

------------------------------------------------------------------------

## 📊 Performance Metrics

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   Confusion Matrix
-   Classification Report

------------------------------------------------------------------------

## 📷 Sample Workflow

``` text
Input Image

      │

Image Preprocessing

      │

CNN Feature Extraction

      │

Feature Classification

      │

Predicted Defect Class
```

------------------------------------------------------------------------

## 📁 Project Structure

``` text
Industrial-Surface-Defect-Detection/

│── dataset/
│── notebooks/
│     └── industrial_surface_defect_detection.ipynb
│── models/
│── results/
│── images/
│── README.md
└── requirements.txt
```

------------------------------------------------------------------------

## 🚀 Installation

``` bash
git clone https://github.com/your-username/Industrial-Surface-Defect-Detection.git
cd Industrial-Surface-Defect-Detection
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Running the Project

Launch Jupyter Notebook:

``` bash
jupyter notebook
```

or open the notebook in Google Colab or Kaggle and run all cells.

------------------------------------------------------------------------

## 📊 Results

The trained model successfully identifies industrial surface defects
with high classification performance.

Outputs include:

-   Training Accuracy
-   Validation Accuracy
-   Loss Curves
-   Confusion Matrix
-   Classification Report

------------------------------------------------------------------------

## 🌟 Applications

-   Steel Surface Inspection
-   Manufacturing Quality Control
-   Automotive Industry
-   Electronics Manufacturing
-   Metal Surface Inspection
-   Smart Factory Automation
-   Industry 4.0 Production Lines

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Real-time defect detection using live camera feeds
-   Deployment using Flask or FastAPI
-   Integration with edge devices
-   Explainable AI using Grad-CAM
-   Object detection with YOLO

------------------------------------------------------------------------

## 👨‍💻 Authors

  Roll Number   Name
  ------------- ---------------
  240108        Rahul Amgoth
  230842        Raju Ramavath
  240411        Akshay Guda

------------------------------------------------------------------------

## 📜 License

This project is developed for academic and educational purposes.

------------------------------------------------------------------------

## ⭐ Acknowledgements

-   Kaggle
-   TensorFlow & Keras
-   OpenCV
-   Google Colab
-   IIT Kanpur

------------------------------------------------------------------------

## 📬 Contact

For suggestions or collaborations, feel free to open an issue or submit
a pull request.

If you found this project useful, don't forget to ⭐ star the
repository!
