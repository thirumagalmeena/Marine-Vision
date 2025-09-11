# MarineVision
# Comparative Analysis of Machine Learning Algorithms for Marine Animal Detection

Marine animal classification is a crucial task in ecological research, biodiversity monitoring, and conservation.  
This project explores **machine learning techniques** to classify marine animals into **five categories**:  
🐬 Dolphin | 🐟 Fish | 🦞 Lobster | 🐙 Octopus | 🐴 Sea Horse  

It compares traditional ML algorithms with deep learning and also provides a **web-based interface** where users can upload an image and see predictions from all models side by side.

---

##  Project Overview
This project has two main components:

1. **Comparative Model Analysis**  
   - Evaluates five machine learning models:  
     -  Random Forest (RF)  
     -  Support Vector Machines (SVM)  
     -  K-Means Clustering  
     -  K-Nearest Neighbors (KNN)  
     -  Convolutional Neural Networks (CNN)  
   - Preprocessing and augmentation are applied to improve robustness.  

2. **Interactive Web Application**  
   - Users can **upload a marine animal image**.  
   - The system runs predictions using **all trained models**.  
   - Displays each model’s output and highlights the **final (correct) classification**.  

---

##  Key Contributions
- Implemented **image preprocessing** (normalization, resizing, augmentation).  
- Benchmarked **traditional ML vs CNNs**, showing CNNs achieve **92% accuracy**.  
- Built a **web app interface** for real-time classification and comparison.  

---
##  Dataset
- **Training images:** 1241  
- **Validation images:** 250  
- **Test images:** 100  
- **Categories:** Dolphin, Fish, Lobster, Octopus, Sea Horse  

Dataset preprocessing included:
- Normalization (standardizing pixel intensity values)  
- Resizing (consistent dimensions)  
- Augmentation (rotation, flipping, brightness adjustments)  

---

##  Methods
### Traditional Machine Learning Models
- **Random Forest (RF):** Decision tree ensemble.  
- **Support Vector Machines (SVM):** Hyperplane-based classification.  
- **K-Means Clustering:** Pattern discovery (unsupervised).  
- **K-Nearest Neighbors (KNN):** Distance-based classification.  

### Deep Learning Model
- **Convolutional Neural Networks (CNN):**  
  - Custom CNN trained on marine animal dataset.  
  - Pre-trained **VGG16** was tested but underperformed on underwater distortions.  

### Web Application
- Developed using **Flask/Streamlit (depending on your setup)**.  
- Upload → preprocess → run through all models → results table.  
- Highlights the **final best prediction (CNN-based)**.  

---

##  Results

| Model                  | Accuracy (%) |
|-------------------------|--------------|
| Random Forest (RF)      |         |
| Support Vector Machines |        |
| K-Means Clustering      | –            |
| K-Nearest Neighbors     |         |
| Convolutional NN (CNN)  | ****    |

**Insights:**  
- CNNs excel at feature extraction and classification.  
- Traditional methods struggle with variability.  
- VGG16 transfer learning was not effective for this dataset.  

---

##  Conclusion
This project demonstrates that **** are the most effective model for marine animal classification.  
The **web app** extends usability, letting users test images and see model comparisons in real time.  

---

##  Installation & Usage
### Requirements
- Python 3.x  
- TensorFlow  
- scikit-learn  
- OpenCV  
- NumPy  
- Matplotlib  
- Flask or Streamlit  

### Setup
Clone the repository:
```bash
git clone https://github.com/your-username/marine-animal-classification.git
cd marine-animal-classification
````

Install dependencies:

```bash
pip install -r requirements.txt
```


##  Web App Features

* Upload an image (`.jpg`, `.png`).
* See predictions from **RF, SVM, KNN, K-Means, CNN**.
* Final “Best Model (CNN)” classification is highlighted.

---

