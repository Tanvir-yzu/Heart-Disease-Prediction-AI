# ❤️ Heart Disease Prediction System

A machine learning-based web application for predicting the likelihood of heart disease using clinical patient data. Built with XGBoost and Streamlit.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Features Explained](#features-explained)
- [Performance](#performance)
- [Disclaimer](#disclaimer)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a heart disease prediction system that uses machine learning to analyze patient clinical data and assess the risk of heart failure. The application provides an easy-to-use web interface for healthcare professionals and researchers to understand the probability of heart disease based on various medical parameters.

---

## ✨ Features

- **Interactive Web Interface**: User-friendly Streamlit-based web application
- **Real-time Predictions**: Instant heart disease risk assessment
- **Confidence Scores**: Provides prediction confidence percentages
- **Risk Level Classification**: Categorizes risk into Low, Moderate, and High
- **Visual Feedback**: Progress bars and animations for better user experience
- **Input Summary**: Displays all input parameters for reference
- **Multilingual Support**: Bengali language interface

---

## 📊 Dataset

The model is trained on heart disease clinical data with the following features:

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Patient's age in years |
| Sex | Categorical | Gender (M/F) |
| ChestPainType | Categorical | Type of chest pain (ATA, NAP, ASY, TA) |
| RestingBP | Numeric | Resting blood pressure in mmHg |
| Cholesterol | Numeric | Serum cholesterol in mg/dl |
| FastingBS | Numeric | Fasting blood sugar > 120 mg/dl (0=No, 1=Yes) |
| RestingECG | Categorical | Resting electrocardiogram results (Normal, ST, LVH) |
| MaxHR | Numeric | Maximum heart rate achieved |
| ExerciseAngina | Categorical | Exercise-induced angina (N=No, Y=Yes) |
| Oldpeak | Numeric | ST depression induced by exercise |
| ST_Slope | Categorical | Slope of the peak exercise ST segment (Up, Flat, Down) |
| HeartDisease | Target | Heart disease presence (0=No, 1=Yes) |

---

## 🧠 Model Architecture

### Algorithm
- **XGBoost Classifier**: An optimized gradient boosting algorithm known for high performance and accuracy on structured/tabular data.

### Pipeline Components
1. **Preprocessing Pipeline**
   - StandardScaler: Normalizes numerical features (Age, RestingBP, Cholesterol, MaxHR, Oldpeak)
   - OneHotEncoder: Encodes categorical features (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope)

2. **Training Configuration**
   - Train-Test Split: 80% training, 20% testing
   - Stratified sampling to maintain class balance
   - Random state: 42 for reproducibility

3. **Model Output**
   - Binary classification (0: No Heart Disease, 1: Heart Disease)
   - Probability scores for risk assessment

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step-by-Step Installation

1. **Clone or download the project**
   ```bash
   cd heart_failure_model
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   streamlit --version
   ```

---

## 💻 Usage

### Option 1: Run the Web Application

1. **Ensure the trained model file exists**
   - The project should include `xgboost_heart_failure_model.pkl`
   - If not, train the model first (see [Model Training](#model-training))

2. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser**
   - Navigate to `http://localhost:8501`

4. **Input patient data**
   - Fill in all the clinical parameters
   - Click the "Predict" button

5. **View results**
   - Prediction: Heart Disease or No Heart Disease
   - Confidence percentage
   - Risk level classification
   - Risk probability metrics

### Option 2: Use the Model Programmatically

```python
import pandas as pd
import joblib

# Load the model
model = joblib.load('xgboost_heart_failure_model.pkl')

# Prepare input data
input_data = {
    'Age': 45,
    'Sex': 'M',
    'ChestPainType': 'ATA',
    'RestingBP': 120,
    'Cholesterol': 200,
    'FastingBS': 0,
    'RestingECG': 'Normal',
    'MaxHR': 130,
    'ExerciseAngina': 'N',
    'Oldpeak': 0.0,
    'ST_Slope': 'Up'
}

input_df = pd.DataFrame([input_data])

# Make prediction
prediction = model.predict(input_df)
probability = model.predict_proba(input_df)

print(f"Prediction: {prediction[0]}")
print(f"Probability: {probability[0]}")
```

---

## 📁 Project Structure

```
heart_failure_model/
│
├── app.py                           # Streamlit web application
├── train_model.py                   # Model training script
├── heart.csv                        # Training dataset
├── xgboost_heart_failure_model.pkl  # Trained model file
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## 🎓 Model Training

### Train a New Model

If you want to retrain the model or use your own dataset:

1. **Prepare your dataset**
   - Place your CSV file as `heart.csv`
   - Ensure it contains all required features

2. **Run the training script**
   ```bash
   python train_model.py
   ```

3. **Output**
   - Model saved as `xgboost_heart_failure_model.pkl`
   - Ready to use with the web application

### Custom Training

Modify `train_model.py` for custom training:

```python
# Adjust train-test split ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Customize XGBoost parameters
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    ))
])
```

---

## 🖥️ Web Application

### Interface Features

1. **Sidebar**: Information about the application and medical disclaimer
2. **Input Form**: Two-column layout for efficient data entry
   - Age, Sex, Chest Pain Type
   - Resting BP, Cholesterol, Fasting Blood Sugar
   - Resting ECG, Max Heart Rate, Exercise Angina
   - Oldpeak (ST Depression), ST Slope
3. **Input Summary**: Expandable section showing all entered values
4. **Prediction Display**:
   - Success/Error messages based on prediction
   - Confidence percentage
   - Visual effects (snow/balloons)
   - Risk level classification
5. **Metrics**: Risk probability and Max Heart Rate display

### User Experience

- **Responsive Design**: Centered layout optimized for different screen sizes
- **Visual Feedback**: Progress bar animation during prediction
- **Animations**: Pulse effect on title, snow/balloon effects for results
- **Clear Indicators**: Color-coded results (green for healthy, red for disease)

---

## 📖 Features Explained

### Medical Parameters

- **Age**: Advanced age is a risk factor for heart disease
- **Sex**: Biological sex can affect heart disease risk
- **ChestPainType**: 
  - ATA: Atypical Angina
  - NAP: Non-Anginal Pain
  - ASY: Asymptomatic
  - TA: Typical Angina
- **RestingBP**: Normal resting blood pressure is typically 90-120 mmHg
- **Cholesterol**: High cholesterol (>200 mg/dl) increases heart disease risk
- **FastingBS**: Fasting blood sugar above 120 mg/dl indicates diabetes risk
- **RestingECG**:
  - Normal: Normal ECG
  - ST: ST-T wave abnormality
  - LVH: Left Ventricular Hypertrophy
- **MaxHR**: Maximum heart rate achieved during exercise
- **ExerciseAngina**: Chest pain during physical activity
- **Oldpeak**: ST depression measured during exercise relative to rest
- **ST_Slope**:
  - Up: Upsloping (better prognosis)
  - Flat: Flat (intermediate)
  - Down: Downsloping (worse prognosis)

---

## 📈 Performance

The XGBoost model provides:
- **High Accuracy**: Optimized for classification accuracy on medical data
- **Fast Inference**: Quick prediction time suitable for real-time applications
- **Probability Outputs**: Confidence scores for risk assessment
- **Robustness**: Handles missing values and categorical variables effectively

---

## ⚠️ Disclaimer

**IMPORTANT MEDICAL DISCLAIMER**

This application is intended for **educational and research purposes only**. It should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment.

- The predictions are based on a machine learning model trained on a specific dataset
- Results may not reflect the full complexity of individual patient conditions
- Always consult with qualified healthcare professionals for medical decisions
- Do not use this tool for emergency medical situations
- The developers assume no liability for decisions made based on this application

---

## 🛠️ Technologies Used

- **Python**: Programming language
- **Streamlit**: Web application framework
- **XGBoost**: Gradient boosting machine learning library
- **scikit-learn**: Machine learning utilities (preprocessing, pipelines)
- **pandas**: Data manipulation and analysis
- **joblib**: Model serialization
- **numpy**: Numerical computing

### Key Dependencies

```
streamlit==1.51.0
xgboost==3.1.1
scikit-learn==1.7.2
pandas==2.3.3
numpy==2.3.5
joblib==1.5.2
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add more detailed model performance metrics
- [ ] Implement data visualization for results
- [ ] Add support for batch predictions
- [ ] Include model interpretability features (SHAP values)
- [ ] Add multi-language support (English, Spanish, etc.)
- [ ] Implement user authentication for healthcare professionals
- [ ] Add data export functionality (PDF reports)

---

## 📝 License

This project is provided as-is for educational purposes. Please ensure you have appropriate rights to use any datasets or models.

---

## 🔗 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Kaggle Heart Disease Dataset](https://www.kaggle.com/fedesoriano/heart-failure-prediction)

---

## 📧 Contact

For questions or suggestions about this project, please open an issue in the repository.

---

**Built with ❤️ for better healthcare awareness**
