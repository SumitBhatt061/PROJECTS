from flask import Flask, render_template, request
import pandas as pd
import pickle
import json

app = Flask(__name__)
model = pickle.load(open('LinearRegressionModel.pkl','rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    # dictionary for company -> models
    car_models_dict = {}
    for company_name in companies:
        car_models_dict[company_name] = sorted(car[car['company']==company_name]['name'].unique())
    car_models_json = json.dumps(car_models_dict)

    return render_template('index.html', companies=companies, years=year,
                           fuel_types=fuel_type, car_models_json=car_models_json)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    if not all([company, car_model, year, fuel_type, driven]):
        return "Please fill all fields"

    year = int(year)
    driven = int(driven)

    prediction = model.predict(pd.DataFrame([[car_model, company, year, driven, fuel_type]],
                                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    return str(round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)

