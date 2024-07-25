from flask import Flask, render_template, request, jsonify
from sqlalchemy import create_engine
import pandas as pd
import pickle
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        user = request.form['user']
        pw = request.form['password']
        db = request.form['database']
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
        try:
            data = pd.read_csv(f)
        except Exception as e:
            try:
                data = pd.read_excel(f)
            except Exception as e:
                return f"Error: {e}"

        numeric_features = data.select_dtypes(exclude=['object']).columns
        processed1 = joblib.load('processed1')
        model = pickle.load(open('Clust_airstat.pkl', 'rb'))
        
        data1 = pd.DataFrame(processed1.transform(data[numeric_features]), columns=numeric_features)
        prediction = pd.DataFrame(model.predict(data1), columns=['cluster_id'])
        prediction = pd.concat([prediction, data], axis=1)
        
        prediction.to_sql('Airline_pred_kmeans', con=engine, if_exists='append', chunksize=1000, index=False)
        
        html_table = prediction.to_html(classes='table table-striped')
        
        return render_template("data.html", Y=html_table)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    model = pickle.load(open('Clust_airstat.pkl', 'rb'))
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
