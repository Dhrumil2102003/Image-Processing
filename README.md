
# 🖼️ Background Removal Using DeepLabV3+ with Streamlit

This project provides an end-to-end solution for **removing backgrounds from images** using a **DeepLabV3+** semantic segmentation model with a ResNet50 backbone. The project includes data preprocessing, training, evaluation, prediction, and an easy-to-use web interface via **Streamlit**.

## 🚀 Features

- 📁 **Data Loader** with preprocessing and augmentation.
- 🧠 **Custom DeepLabV3+ model** using ResNet50 backbone.
- 🧪 **Training pipeline** with advanced callbacks and metrics.
- 📊 **Model evaluation** with accuracy, IoU, Dice, Precision, Recall, F1-score.
- 🎯 **Prediction script** for new unseen images.
- 🌐 **Streamlit Web App** to interactively remove background and download results.


## 📂 Project Structure

.
├── app.py             # Streamlit app for background removal

├── data.py            # Data loader and augmentation logic

├── eval.py            # Evaluation and metric reporting

├── metrics.py         # Custom metrics: IoU, Dice loss, Dice coefficient

├── model.py           # DeepLabV3+ model definition

├── predict.py         # Background prediction on test images

├── train.py           # Training pipeline

├── files/             # Folder to save models, logs, evaluation scores

├── new_data/          # Augmented and resized image/mask dataset

└── test_images/       # Folder to run predictions on new images

## 🏗️ How It Works

### 1. **Data Preparation**
- Input images and masks are loaded and augmented (`data.py`).
- Augmentations include: horizontal flip, grayscale conversion, dropout, rotation, etc.
- Output is center-cropped or resized to 256x256.

### 2. **Model Architecture**
- The model (`model.py`) is a customized **DeepLabV3+** with:
  - ASPP (Atrous Spatial Pyramid Pooling)
  - Squeeze-and-Excitation blocks
  - ResNet50 pretrained encoder

### 3. **Training**
- Train on augmented data using Dice loss and evaluation metrics (`train.py`).
- Includes callbacks: model checkpointing, early stopping, TensorBoard, learning rate reduction.

### 4. **Evaluation**
- Evaluate on test dataset and save composite images showing input, ground truth, prediction, and masked output (`eval.py`).
- Saves per-image and average scores in `files/score.csv`.

### 5. **Prediction**
- Run inference on new test images and visualize background removal (`predict.py`).

### 6. **Streamlit App**
- Upload an image.
- View original and background-removed result.
- Download the final output (`app.py`).


## 🧪 Requirements

Install dependencies via:

  pip install -r requirements.txt

Key Libraries:
- TensorFlow
- OpenCV
- Albumentations
- Streamlit
- scikit-learn
- tqdm
- Pillow

## 🖥️ Running the App

### Train the Model
  python train.py

### Evaluate the Model
  python eval.py

### Predict on Test Images
  python predict.py

### Launch Streamlit Web App
  streamlit run app.py


📌 Notes

- Image input size is fixed to **256x256** (can be changed in code).
- Model weights are saved in `files/model.h5` and later used in prediction/app.
- Ensure the folder structure exists before running scripts.

