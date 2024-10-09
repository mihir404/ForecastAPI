import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
import numpy as np

app = Flask(__name__)

# Dummy historical sales data for training the model
data = {
    'product_id': [1, 2, 3, 4, 5],
    'historical_sales_qty': [100, 200, 300, 400, 500],
    'historical_revenue': [1000, 2000, 3000, 4000, 5000],
    'demand': [150, 250, 350, 450, 600]
}

df = pd.DataFrame(data)
X = df[['historical_sales_qty', 'historical_revenue']]
y = df['demand']

model = RandomForestRegressor()
model.fit(X, y)

@app.route('/predict-demand', methods=['POST'])
def predict_demand():
    input_data = request.json
    predictions = []

    for product in input_data:
        product_id = product.get('product_id')
        historical_sales_qty = product.get('historical_sales_qty')
        historical_revenue = product.get('historical_revenue')

        features = np.array([[historical_sales_qty, historical_revenue]])
        predicted_demand = model.predict(features)

        predictions.append({
            'product_id': product_id,
            'predicted_demand': predicted_demand[0]
        })

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
