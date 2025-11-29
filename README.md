# SlideLab AI — NTD Vision

**Track:** Health & Public Safety  

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://slide-ai.streamlit.app/)

SlideLab AI is an AI-powered microscopy tool that assists healthcare workers in detecting malaria from stained blood-smear images. The system uses a deep learning model to classify cells as **parasitized** or **uninfected**, aiming to support diagnosis in resource-limited settings.

---

## Team

- **Adeniyi Micheal Oluwafemi**  
  Fellow ID: FE/23/44017546  
  Role: Team Lead  
  Contact: oluwafemiadeniyi772@gmail.com  

- **Ifeanyi Hyacint Muotoe**  
  Fellow ID: FE/25/9928453022  

- **Olutunde Stephen Anuoluwa**  
  Fellow ID: FE/23/85039993  

---

## Overview

Malaria diagnosis in many regions still relies heavily on manual microscopy, which is time-consuming and depends on the skill and availability of trained personnel. SlideLab AI provides an AI-assisted second opinion by analysing digitised blood-smear images and returning a prediction in real time.

The current version focuses on malaria, but the architecture is designed to be extended to other neglected tropical diseases (NTDs) in the future.

---

## Features

- **AI-assisted slide analysis**  
  Upload a stained blood-smear image and receive an instant prediction.

- **Binary malaria classification**  
  Classifies images into two classes: **parasitized** and **uninfected**.

- **Visual feedback**  
  Displays the uploaded image together with the predicted class and probability.

- **Extensible design**  
  The pipeline can be adapted to support additional NTDs such as filariasis and loiasis.

---

## Dataset

The model is trained on a publicly available malaria blood-smear image dataset from the  
**Lister Hill National Center for Biomedical Communications (LHNCBC)** at the  
**U.S. National Library of Medicine**.

- Thin and thick blood-smear microscopy images  
- Expert-annotated labels for parasitized vs. uninfected cells  
- Anonymised images collected under real clinical conditions  

This dataset provides realistic examples of what diagnostic labs in malaria-endemic regions encounter.

---

## Model

- **Architecture:** EfficientNetB0  
- **Input size:** 180 × 180 pixels  
- **Number of classes:** 2 (parasitized, uninfected)  
- **Preprocessing:** `tf.keras.applications.efficientnet.preprocess_input`  

During development, standard evaluation tools (confusion matrix, ROC curve, probability histograms) were used to assess performance and calibration.

---

## Project Structure

Current repository layout:

- `Malaria_Cell_Classification_Model.h5`  
  Trained malaria cell classification model.

- `Malaria_Cell_Classification_Model_Notebook.ipynb`  
  Jupyter notebook used for model training, experiments, and analysis.

- `StreamlitWebApp.py`  
  Main Streamlit application used for running the web interface and model inference.

- `requirements.txt`  
  Python dependencies required to run the project.

- `README.md`  
  Project documentation (this file).

- `LICENSE`  
  MIT license for this project.

---

## How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/OfemiAdeniyi/malaria-app.git
   cd malaria-app
   ```

2. **Create and activate a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Start the Streamlit app**

   ```bash
   streamlit run StreamlitWebApp.py
   ```

5. **Use the app**

   - Open the URL shown in the terminal (usually `http://localhost:8501`).
   - Upload a stained blood-smear image.
   - View the predicted class (parasitized or uninfected) and the associated probability.

---

## Limitations and Disclaimer

- The current version is trained and evaluated on a specific public dataset and may not cover all slide preparation techniques or imaging conditions.
- This tool is intended for **research and demonstration purposes only**.  
  It is **not** a certified medical device and should not be used as a standalone basis for clinical decision-making.

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
