Face Recognition Pipeline
A complete pipeline for face detection, face extraction, feature extraction, data augmentation, and training a VGG16-based classifier for face recognition. This project provides scripts to:

Process videos and detect faces using a Caffe-based DNN model
Save detected faces as JSON files
Convert JSON face data into DataFrames
Crop and save faces from original videos
Compute embeddings for known actors and match new faces with these embeddings
Augment data to increase dataset size
Train a VGG16 model on the final dataset
(Optional) Predict the identity of new faces using the trained model
Table of Contents
Overview
Project Structure
Data Folder Structure
Usage
Modules Description
Contact
Overview
Video Processing & Face Detection: Uses a Caffe-based Deep Neural Network to detect faces in video frames.
JSON Generation: Saves bounding-box coordinates of detected faces in JSON files.
DataFrame Creation: Converts JSON data into Pandas DataFrames for further processing.
Face Extraction: Crops faces from original videos using bounding-box coordinates.
Feature Extraction & Matching: Generates embeddings for known actors, matches new faces, and organizes them accordingly.
Data Augmentation: Increases dataset size by applying transformations (e.g., flips, rotations).
Model Training: Finetunes a pre-trained VGG16 network to classify faces.
Prediction (Optional): Loads the trained model and predicts identities for new face images.
Project Structure
bash
Copy code
├── config.py               # Configuration file (paths, hyperparameters, etc.)
├── main.py                 # Main orchestrator script
├── modules                 # Directory containing all modules
│   ├── face_detection.py
│   ├── json_to_dataframe.py
│   ├── face_extraction.py
│   ├── feature_extraction.py
│   ├── data_augmentation.py
│   ├── model_training.py
│   └── prediction.py
├── README.md               # This README file
├── requirements.txt        # Python dependencies (optional)
└── ...
config.py: Central configuration of paths, thresholds, hyperparameters, etc.
main.py: Ties together all steps in the face recognition pipeline.
modules/: Contains specialized scripts (detection, extraction, augmentation, training, etc.).
Data Folder Structure
bash
Copy code
data/
├── videos/                   # Contains the input video files (.mp4, .avi, etc.)
├── jsons/                    # Stores JSON files with face bounding-box data
├── faces/                    # Face crops extracted from videos
│   └── video_name/           # Each video has a corresponding folder with cropped faces
├── actors_ready/             # Labeled images of known actors
│   └── ActorName/
│       └── image1.jpg
├── augmented/                # Stores augmented images (after data augmentation)
└── dataset/                  # Final dataset in ImageFolder format for training
    ├── class1/ (Actor A)
    ├── class2/ (Actor B)
    └── ...
videos/: Original raw videos.
jsons/: Outputs of the face detection step (bounding boxes in JSON).
faces/: Cropped faces from videos, organized by each video filename.
actors_ready/: Reference images for known actors used for generating embeddings.
augmented/: Images produced via data augmentation.
dataset/: Final dataset for training (in PyTorch’s ImageFolder structure).
Usage
Configure Paths:
Open config.py and set the following variables according to your setup:

python
Copy code
VIDEO_FOLDER = "path/to/your/video/files"
JSON_FOLDER = "path/to/save/json/files"
FACES_FOLDER = "path/to/save/cropped/faces"
ACTORS_READY_FOLDER = "path/to/known/actors/images"
AUGMENTED_FOLDER = "path/to/save/augmented/data"
DATASET_DIR = "path/for/final/training/dataset"
# etc.
Run the Pipeline:

bash
Copy code
python main.py
This will:

Load the Caffe face detection model
Detect faces in the videos
Convert face detection results into JSON files
Crop face images and store them
Generate embeddings for known actors and match new faces
Augment images if needed
Train a VGG16 model on the final dataset
Save the fine-tuned model as fine_tuned_vgg16.pth
Check Outputs:

JSON files in JSON_FOLDER
Cropped faces in FACES_FOLDER
Augmented images in AUGMENTED_FOLDER
Trained model fine_tuned_vgg16.pth in the project directory
Modules Description
face_detection.py
Loads a Caffe model for face detection.
Processes videos frame by frame (with a skip interval).
Saves bounding box coordinates to JSON (after Non-Max Suppression).
json_to_dataframe.py
Converts JSON face detections into Pandas DataFrames.
Prepares data for cropping faces and further processing.
face_extraction.py
Crops faces from videos using bounding-box data in the DataFrames.
Saves the cropped face images locally.
feature_extraction.py
Generates embeddings for known actors (e.g., with a pretrained network).
Matches new faces with known actor embeddings.
Organizes matched faces in specific folders.
data_augmentation.py
Augments face images via various transformations (flips, rotations, etc.).
Increases the dataset size to help prevent overfitting.
model_training.py
Creates PyTorch Datasets/DataLoaders from an ImageFolder structure.
Loads a pre-trained VGG16, replaces the final layer, and trains the network.
Saves the fine-tuned model.
prediction.py (Optional)
Loads the saved VGG16 model.
Predicts the identity of faces by performing inference on cropped images.
Contact
For any questions, suggestions, or feedback, feel free to open an issue or reach out:

Isar Shmueli

LinkedIn
GitHub: isarinio22
