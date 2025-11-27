### SlideLab AI — NTD Vision

Track: Health & Public Safety
Team: 
1. Adeniyi Micheal Oluwafemi, Fellow ID: FE/23/44017546
2. Ifeanyi Hyacint Muotoe, Fellow ID: FE/25/9928453022
3. Olutunde Stephen Anuoluwa, Fellow ID:FE/23/85039993
   
*Lead: Adeniyi Micheal Oluwafemi*
Contact: oluwafemiadeniyi772@gmail.com

### Overview

SlideLab AI is an AI-powered microscopy platform designed to assist healthcare providers in diagnosing malaria and other neglected tropical diseases (NTDs) through blood-smear image analysis. Built with EfficientNetB0 and advanced preprocessing, the system provides accurate predictions in real-time, helping bridge gaps in diagnostics, especially in resource-limited regions.

### Features

1. AI-assisted slide analysis: Upload a stained blood-smear image to get instant predictions.
2. Classifies parasitized vs. uninfected cells (currently for malaria).
3. Visual feedback: See uploaded cell images and predicted probabilities.
4. Debug mode: Inspect preprocessing ranges for transparency and reproducibility.
5. Scalable: Designed to include other NTDs such as filariasis and loiasis.

### Dataset Description
We used the publicly available malaria blood-smear image dataset from the Lister Hill National Center for Biomedical Communications (LHNCBC) at the U.S. National Library of Medicine. This dataset contains thin and thick blood-smear microscopy images from both infected and uninfected patients, with expert-annotated labels identifying parasitized and healthy red blood cells. The images are anonymized and collected under real clinical conditions, making them highly representative of what diagnostic labs encounter in malaria-endemic regions. By leveraging this dataset, SlideLab AI trained and evaluated AI models to accurately classify blood-smear images, distinguishing infected cells from healthy ones and enabling reliable AI-assisted malaria diagnostics.

### Model

Architecture: EfficientNetB0

Classes: parasitized, uninfected

Input size: 180×180 pixels

Preprocessing: tf.keras.applications.efficientnet.preprocess_input

<img width="400" height="300" alt="confusion_matrix" src="https://github.com/user-attachments/assets/a50c34f1-8181-40fa-9d4d-8d6fcfe9382f" />

<img width="400" height="300" alt="probabilities_histogram" src="https://github.com/user-attachments/assets/bd2988e1-5eea-4a5f-90f0-785e96ba66b8" />

<img width="400" height="300" alt="roc_curve" src="https://github.com/user-attachments/assets/3dd76a57-c748-478e-90b1-adc46832118c" />


### Impact

SlideLab AI empowers healthcare workers by providing rapid, accurate AI diagnostics, potentially improving early detection of malaria and NTDs, especially in regions with limited laboratory resources.
