# House Price Prediction 

This is a Machine Learning project that predicts house prices using the Boston Housing Dataset.

I built this project using Python and different Machine Learning algorithms. After comparing the models, Random Forest Regressor was selected for the final prediction.

## About the Project

The model takes different house-related features as input, such as:

- Crime rate
- Number of rooms
- Age of the house
- Property tax
- Highway accessibility
- Distance to employment centres
- Pupil-teacher ratio
- Other related features

The predicted price is then shown through a simple Streamlit web application.

## Machine Learning Models Used

I tried the following models:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

Random Forest gave the best overall performance, so it was used as the final model.

## Model Performance

The final Random Forest model achieved:

- Test RMSE: **2.8082**
- Cross Validation RMSE: **3.6657**

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Jupyter Notebook
- Joblib

## Project Files

- `House_Price_Prediction.ipynb` - Jupyter Notebook containing the complete ML process
- `app.py` - Streamlit web application
- `housing.data` - Dataset
- `housing.name` - Dataset information
- `house_price_model.pkl` - Trained Random Forest model
- `house_price_pipeline.pkl` - Data preprocessing pipeline

## How to Run

First install the required libraries:

```bash
pip install pandas numpy scikit-learn streamlit joblib

Then run the Streamlit application:
streamlit run app.py

The application will open in the browser.

## Result

The project provides a simple interface where users can enter house details and get an estimated house price using the trained Machine Learning model.

## Author

  Tanu2004-st





