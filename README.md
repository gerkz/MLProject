# MLProject
INF 2008 Machine Learning

# User Manual

## 1. Environment Setup

Follow these steps to set up the environment for running the project:

### 1.1 Clone the Repository
Start by cloning the repository to your local machine:
git clone <repository-url>

### 1.2 Navigate to the Project Directory
cd <project-directory>

### 1.3 Create a Virtual Environment
python3 -m venv venv (mac)
source venv/bin/activate (mac)

python -m venv venv (win)
.\venv\Scripts\activate (win)

### 1.4 Install Dependencies
pip install -r requirements.txt

# 2. Data Preprocessing for Signature Recognition

## Overview of Data Preprocessing

The goal of this preprocessing step is to improve the quality of signature images and generate balanced signature pairs that will be used for model training. The process includes several stages: image enhancement, signature pairing, feature extraction, and normalization.

### 1. Enhancing Image Quality  
The first step is to enhance the signature images to extract relevant features:
- **Convert to Grayscale:** Images are converted to grayscale and converted to `uint8` format for consistent processing.
- **Gaussian Blur:** Applied to reduce noise while maintaining important signature details.
- **Canny Edge Detection:** Used to highlight signature strokes, which are crucial for feature extraction.
- **Otsu’s Thresholding:** Converts images into binary format to improve contrast and clarity.

### 2. Visualizing Before and After Pre-Processing  
Before diving into the feature extraction process, it's useful to visualize the effect of preprocessing:
- **Original Images:** These images are displayed to observe their raw quality.
- **Processed Images:** The processed images are then shown, allowing us to compare improvements in contrast, edge detection, and noise reduction. This step ensures that the features extracted later will be clearer and more relevant for training.

### 3. Generating Signature Pairs  
Once the images are preprocessed, we generate pairs of signatures for comparison:
- **Genuine-Genuine (G-G) Pairs:** These pairs are created from signatures of the same individual.
- **Genuine-Forged (G-F) Pairs:** These pairs are formed by matching genuine signatures with their corresponding forgeries.
- **Balanced Dataset:** To ensure balanced training, the number of G-G and G-F pairs are kept equal.  
  - **G-G Pairs:** 15,180  
  - **G-F Pairs:** 15,180  
  - **Total Pairs:** 30,360  

This process ensures that we have a well-balanced dataset to train the model, containing both genuine and forged signatures for accurate learning.

## Feature Extraction for Signature Recognition

Once the signature pairs are ready, we extract **74 features** from each signature to enable the model to distinguish between genuine and forged signatures. These features are extracted across three key categories:

### 1. **Edge-Based Features (48 features)**
- **Sobel Gradients:** Detects edges, capturing both stroke strength and direction.
- **Histograms:** Generates 32-bin edge magnitude and 16-bin edge orientation histograms to quantify edge distribution.

### 2. **Texture-Based Features (Gabor Filters) (24 features)**
- **Gabor Filters:** Applied at 3 different frequencies and 4 orientations to analyze texture patterns in the signature.
- **Mean and Variance:** These are extracted for each combination of Gabor filters, resulting in 24 texture features.

### 3. **Intensity-Based Features (2 features)**
- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Enhances contrast in the images.
- **Shannon Entropy and Variance:** These measures help analyze pixel intensity distribution and complexity.

These 74 features provide a rich, detailed representation of the signatures, enabling effective differentiation between genuine and forged signatures.

## Feature Normalization

To ensure that the features are on consistent scales and are comparable, we apply different normalization techniques tailored to the type of features:

### 1. **Edge-Based Features (Edge Magnitude & Orientation) → MinMax Scaling**
- Scales the values between 0 and 1, preserving the edge intensity proportions.

### 2. **Texture-Based Features (Gabor Responses) → Standard Scaling**
- Centers values around 0 with a standard deviation of 1 to improve texture analysis.

### 3. **Intensity-Based Features (CLAHE Entropy & Variance) → Robust Scaling**
- Uses median and interquartile range, making it resistant to outliers.

By applying these scaling techniques, we ensure that all features have consistent ranges and distributions, preventing any one feature type from dominating the training process.

## Dataset Processing

With the features extracted and normalized, the next step is to prepare the dataset for training. This involves:

### 1. **Preparing Data for Training**
- **Preprocess Images:** Apply contrast enhancement, edge detection, and thresholding.
- **Generate Signature Pairs:** Create both genuine-genuine and genuine-forged pairs for model training.
- **Feature Extraction:** Extract edge, texture, and intensity features and normalize them.
- **Shuffle Data:** Shuffle the dataset to avoid any ordering bias.

### 2. **Loading Signature Images**
- Load genuine and forged signatures from the dataset.
- Invert pixel values and resize images to 200×150 pixels.
- Organize signatures into genuine and forged arrays.

### 3. **Final Dataset**
- The processed dataset contains 30,360 pairs with 74 features each.
- Labels: genuine (1) or forged (0).
- Saved as `X.npy` and `y.npy` for model training.

This pipeline ensures an optimized dataset for training.

## Feature Trend and Pattern Analysis

To gain a better understanding of the **overall trends and patterns** across the three feature categories: **Edge**, **Texture**, and **Intensity**, we analyze these features computed from both genuine and forged signatures. This analysis helps capture key differences in signature characteristics.

### Comparison of Genuine and Forged Signatures
- **Visualization:** Side-by-side bar charts are used to compare feature distributions between genuine and forged signatures.
- **Feature Categories:** 
  - **Edge:** Differences in edge magnitude.
  - **Texture:** Variations in texture complexity.
  - **Intensity:** Distribution of intensity across signatures.
  
While this analysis does not directly influence feature selection for model training, it provides valuable insights into how different features help distinguish genuine signatures from forgeries.

## 3. Experiment Design Plan

The experiment evaluates the effectiveness of different machine learning models in detecting signature forgery using the CEDAR Signature paired dataset. The dataset will be split into 70% for training and 30% for validation. A custom test set, consisting of 11 individuals' signatures (20% of the dataset), will be used for final testing.

### Models Evaluated
Three models will be trained and compared:
- **Support Vector Machine (SVM):** Effective in high-dimensional spaces, ideal for binary classification.
- **Random Forest:** Ensemble learning method combining decision trees to reduce overfitting.
- **XGBoost:** A gradient boosting technique known for its efficiency and scalability in capturing complex patterns.

These models will be evaluated across four phases:
1. **Baseline Model Training:** Train each model with default hyperparameters to establish baseline performance.
2. **Hyperparameter Optimization (Grid Search):** Fine-tune hyperparameters for each model to improve performance.
3. **Scaler Selection:** Experiment with different feature scaling techniques to identify the most effective one.
4. **Final Model Evaluation:** Assess the best-performing models on the validation set to ensure generalization.

The models will be compared based on **performance metrics** (accuracy, precision, recall, F1-score) and **computational metrics** (model size, training/testing time, and memory usage). The top models will then be tested on the real-world dataset for final validation.

Finally, the best-performing classical model will be compared with a deep learning model (Siamese Network) to evaluate the trade-offs in accuracy, efficiency, and interpretability.