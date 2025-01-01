This repository provides a complete pipeline for face detection, face extraction, feature extraction, data augmentation, and training a VGG16-based classifier for face recognition. The project includes scripts to process videos, detect faces, convert detection results to DataFrames, crop face images, augment data, and train a neural network model.
1.Overview
2.Video Processing & Face Detection: Uses a Caffe-based Deep Neural Network model to detect faces in video frames.
3.JSON Generation: Saves bounding box coordinates of detected faces as JSON files.
4.DataFrame Creation: Converts JSON face data into DataFrames for easier manipulation.
5.Face Extraction: Crops and saves faces from original videos using bounding box data.
6.Feature Extraction & Matching: Computes embeddings for known actors, matches new faces with these embeddings, and organizes them accordingly.
7.Data Augmentation: Increases dataset size by applying transformations to existing images.
8.Model Training: Finetunes a pre-trained VGG16 network to classify faces.
9.Prediction (Optional): Provides a script to load the trained model and predict the identity of new faces.

Project Structure
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

data/
├── videos/                   # Contains the input video files (.mp4, .avi, etc.)
├── jsons/                    # Stores JSON files with face bounding-box data
├── faces/                    # Face crops extracted from videos
│   └── video_name/           # Each video has a corresponding folder with cropped faces
├── actors_ready/             # Labeled images of known actors
│   └── ActorName/
│       └── image1.jpg        # Example images per actor
├── augmented/                # Stores augmented images (after data augmentation)
└── dataset/                  # Final dataset for training in ImageFolder structure
    ├── class1/ (Actor A)
    ├── class2/ (Actor B)
    └── ...

    videos/: Original raw videos.
jsons/: Outputs of face detection step, where bounding boxes are serialized.
faces/: Cropped faces from videos, organized by video filename.
actors_ready/: Reference images for known actors that are used to generate embeddings.
augmented/: Augmented images to enrich the dataset.
dataset/: Final ready-to-train dataset, typically structured in subfolders by class for PyTorch’s ImageFolder.

Usage
Configure Paths: Open config.py and set:

VIDEO_FOLDER = Path to your video files
JSON_FOLDER = Where JSON files of detected faces will be saved
FACES_FOLDER = Where cropped face images will be stored
ACTORS_READY_FOLDER = Reference images for known actors
AUGMENTED_FOLDER = Where augmented images will be created
DATASET_DIR = Path for final dataset (ImageFolder format)
(etc.)


RUN THE PIPELINE
python main.py

This will:

Load the Caffe face detection model.
Detect faces in the videos.
Convert detection results (JSON) to DataFrames.
Crop face images and store them.
Generate embeddings for known actors and match them with newly detected faces.
Augment the dataset if needed.
Train a VGG16 model on the final dataset.
Save the fine-tuned model as fine_tuned_vgg16.pth.
Check Outputs:

JSON files in JSON_FOLDER
Cropped faces in FACES_FOLDER
Augmented images in AUGMENTED_FOLDER (if augmentation is performed)
Trained model named fine_tuned_vgg16.pth in your project directory

Modules Description
face_detection.py

Loads a pre-trained Caffe model for face detection.
Processes videos, extracts frames at intervals (frame_skip), detects faces, and applies Non-Max Suppression.
Saves bounding box coordinates to JSON.
json_to_dataframe.py

Converts JSON face detections into Pandas DataFrames.
Prepares data for face cropping and further processing.
face_extraction.py

Uses bounding box coordinates to crop faces from videos.
Saves cropped faces as .png images.
feature_extraction.py (Not fully shown here, but implied from main.py)

Generates embeddings for known actors (e.g., using a pretrained model).
Matches new faces with known actor embeddings and moves matched faces to specific folders.
data_augmentation.py

Augments existing images (random flips, rotations, etc.) to improve model generalization.
model_training.py

Creates PyTorch Datasets and DataLoaders from an ImageFolder structure.
Loads a pre-trained VGG16, replaces the classifier’s final layer, and trains the network.
Saves the trained model.
prediction.py (Optional usage)

Loads the saved fine-tuned model.
Predicts the identity of faces by performing inference on cropped images.

For any questions, suggestions, or feedback, feel free to open an issue or reach out:

Isar Shmueli – [https://www.linkedin.com/in/isar-shmueli-906a60262/](https://www.linkedin.com/in/isar-shmueli-906a60262/)
GitHub: isarinio22

