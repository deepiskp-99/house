from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

data = pd.read_csv("House_Pricing.csv")
data.columns = data.columns.str.strip()
data = data.dropna()  # Remove missing values if any

X = data[['No of Bedrooms','No of Bathrooms','Flat Area (in Sqft)']]
y = data['Sale Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    No_of_Bedrooms = float(request.form['No of Bedrooms'])
    No_of_Bathrooms = float(request.form['No of Bathrooms'])
    flat_area_in_sqft = int(request.form['Flat Area (in Sqft)'])
    prediction = model.predict(np.array([[No_of_Bedrooms, No_of_Bathrooms, flat_area_in_sqft]]))
    output = round(prediction[0],2)
    return render_template('index.html', prediction_text=f"Predicted House Price: ₹{output}")

if __name__ == "__main__":
    app.run(debug=True)
