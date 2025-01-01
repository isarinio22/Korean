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
