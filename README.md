# House Price Prediction Project

## Objective

This project aims to develop a complete machine learning pipeline capable of predicting residential property prices based on key property characteristics and available amenities. The final model is integrated into an interactive web application, allowing users to estimate house prices in real time.

## Dataset

The project uses the **Housing Prices Dataset**, which consists of 545 records and 13 columns, including 12 input features and one target variable (`price`). The dataset contains numerical attributes such as area, number of bedrooms, and bathrooms, along with categorical features representing amenities like air conditioning, guest rooms, and other property facilities.

## Technologies Used

* **Programming Language:** Python 3
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Web Framework:** Streamlit
* **Model Serialization:** Joblib

## Machine Learning Models

The following regression algorithms were trained and compared to determine the most effective model:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor (chosen as the final model after hyperparameter optimization using GridSearchCV)

## Folder Structure

```text
House_Price_Prediction_Project/
├── Model/
│   ├── preprocessing.py
│   ├── train.py
│   ├── best_model.pkl
│   └── requirements.txt
├── Streamlit_App/
│   ├── app.py
│   └── assets/
├── Dataset/
│   └── Housing.csv
├── Report/
│   ├── Project_Report.pdf
│   └── Project_Report.tex
├── Presentation/
│   └── House_Price_Presentation.pptx
├── Demo/
│   └── Demo_Video.mp4
├── README.md
└── requirements.txt
```

## Installation

1. Extract the project files to your preferred location.
2. Open a terminal and move to the project's root directory.
3. Create and activate a Python virtual environment.
4. Install all required packages by running:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

To train the machine learning model from the beginning, execute:

```bash
python Model/train.py
```

## Launching the Web Application

Start the Streamlit application with the following command:

```bash
streamlit run Streamlit_App/app.py
```

## Author

**Akshat Kumar**
