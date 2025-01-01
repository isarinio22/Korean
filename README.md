# Face Recognition Pipeline

A complete pipeline for **face detection**, **face extraction**, **feature extraction**, **data augmentation**, and **training** a VGG16-based classifier for face recognition.

---

## Table of Contents
1. [Overview](#overview)  
2. [Project Structure](#project-structure)  
3. [Data Folder Structure](#data-folder-structure)  
4. [Usage](#usage)  
5. [Modules Overview](#modules-overview)  
6. [Contact](#contact)

---

## Overview

1. **Video Processing & Face Detection**  
   - Uses a Caffe-based DNN model to detect faces in video frames.  
2. **JSON Generation**  
   - Stores bounding-box coordinates of detected faces in JSON files.  
3. **DataFrame Creation**  
   - Converts JSON detections into Pandas DataFrames for further manipulation.  
4. **Face Extraction**  
   - Crops faces from videos using bounding-box data.  
5. **Feature Extraction & Matching**  
   - Generates embeddings for known actors and matches new faces with these embeddings.  
6. **Data Augmentation**  
   - Applies transformations (flip, rotate, etc.) to increase dataset size.  
7. **Model Training**  
   - Finetunes a pre-trained VGG16 model for face classification.  
8. **Prediction (Optional)**  
   - Loads the trained model and predicts the identity of new faces.

---

## Project Structure

├── config.py # Main configuration (paths, hyperparams, etc.) ├── main.py # Orchestrates the entire pipeline ├── modules # Directory containing all modules │ ├── face_detection.py │ ├── json_to_dataframe.py │ ├── face_extraction.py │ ├── feature_extraction.py │ ├── data_augmentation.py │ ├── model_training.py │ └── prediction.py ├── README.md # This README ├── requirements.txt # Project dependencies (optional) └── ...


- **`config.py`** – Set your directory paths, thresholds, hyperparameters, etc.  
- **`main.py`** – High-level script that runs each step of the pipeline.  
- **`modules/`** – Specialized Python scripts for detection, extraction, augmentation, training, etc.

---

## Data Folder Structure

data/ ├── videos/ # Input videos (.mp4, .avi, etc.) ├── jsons/ # JSON files storing face bounding-box data ├── faces/ # Cropped faces from each video │ └── video_name/ ├── actors_ready/ # Known actors' images (for embedding) │ └── ActorName/ │ └── image1.jpg ├── augmented/ # Augmented versions of images └── dataset/ # Final dataset (PyTorch ImageFolder structure) ├── class1/ ├── class2/ └── ...


- **`videos/`** – Original videos for face detection.  
- **`jsons/`** – Detected bounding boxes per frame, in JSON format.  
- **`faces/`** – Cropped face images from each video.  
- **`actors_ready/`** – Reference images for known actors.  
- **`augmented/`** – Images after augmentation.  
- **`dataset/`** – Training dataset in [ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) format.

---

## Usage

1. **Configure**  
   - Update `config.py` with your folder paths, thresholds, and other parameters.  

2. **Install Requirements**  
   ```bash
   pip install -r requirements.txt

   
Run the Pipeline
python main.py

This will:

Load the Caffe face detection model
Detect faces & generate JSON files
Convert JSON to Pandas DataFrames
Crop and organize face images
Build embeddings for known actors and match them with new faces
Augment the dataset if needed
Train the VGG16 model
Save the model as fine_tuned_vgg16.pth
Check Outputs

JSON files in jsons/
Cropped faces in faces/
Augmented images in augmented/ (if performed)
Trained model (fine_tuned_vgg16.pth) in the project directory

Modules Overview
face_detection.py
Loads a Caffe model, processes videos to detect faces, and saves bounding boxes to JSON.

json_to_dataframe.py
Converts JSON bounding boxes into Pandas DataFrames for organized face data.

face_extraction.py
Crops faces from videos using bounding-box data and saves them as images.

feature_extraction.py
Generates embeddings for known actors and matches new faces with these embeddings.

data_augmentation.py
Applies transformations (flip, rotate, etc.) to augment and expand the dataset.

model_training.py
Finetunes a pre-trained VGG16 on the prepared dataset and saves the trained model.

prediction.py
Loads the trained model for prediction on new face images.

Contact
Isar Shmueli

LinkedIn
GitHub
