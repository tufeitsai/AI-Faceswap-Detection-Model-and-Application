# **AI-Faceswap Detection Models and Application**
USC CSCI 566 Final Project

This repository contains our project for detecting AI face-swap images using self-trained models and a simple web application. The goal is to identify whether an image is real or AI-generated, using a combination of self-collected datasets and cutting-edge deep learning techniques.

Demo Video: https://youtu.be/5K0oJ6-KKbM

---

## **Dataset**
We created our dataset by:
- Collecting **real images** from web scraping.
- Generating **AI face-swap images** using outdated AI face-swap models, including **Vindoz**, **Akool**, **Simpswap**, and **Face2Face**.

---

## **Models**
We trained three different models to handle AI face-swap detection:
1. **Vision Transformer (ViT):**
   - Trained on original images.
2. **CNN with Xception (Pixel Noise):**
   - All images were pre-processed to pixel noise representations.
   - Used this dataset to train the CNN model.
3. **CNN with Xception (Face Edge Enhance):**
   - All images were pre-processed to highlight face edges using OpenCV.
   - This dataset was used to train the CNN model.

Each model predicts whether an uploaded image is **real** or **fake**.

---

## **Web Application**
We developed a simple web application using **Flask** that allows users to:
1. Upload a picture.
2. Select one of the three models (Vision Transformer, Pixel Noise CNN, or Face Edge Enhanced CNN).
3. Get predictions (output as **Real** or **Fake**).

---

## **Model Files**
Due to size limitations, the trained model files are too large to upload directly to GitHub. You can download the models from the following Google Drive link:  
[Download Model Files](https://drive.google.com/drive/folders/167MZu8ox3nET96PjM7RPqfIC45CGZHgh?usp=sharing)

---

## **Setup Instructions**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/AI-Faceswap-Detection.git
   cd AI-Faceswap-Detection

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
3. Download the models from the provided Google Drive link and place the "Models" folder into the complete folder
4. Run the web application:
     ```bash
     python app.py

5. Open your browser and go to http://127.0.0.1:5000/.

