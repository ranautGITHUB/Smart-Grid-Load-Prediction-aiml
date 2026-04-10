from flask import Flask, render_template, request, redirect, url_for
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

try:
    data_path = os.path.join(BASE_DIR, "data.csv")
    data = pd.read_csv(data_path)

    numeric_data = data.select_dtypes(include=['number'])

    X = numeric_data.iloc[:, 0].values.reshape(-1,1)
    y = numeric_data.iloc[:, 1].values

    X = X[:100]
    y = y[:100]

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    pred_all = model.predict(X)
    accuracy = round(r2_score(y, pred_all) * 100, 2)

except Exception as e:
    print("ERROR IN DATA:", e)
    X = np.array([[1],[2],[3]])
    y = np.array([10,20,30])
    accuracy = 0
    model = None

def create_graph(x, y, pred_x=None, pred_y=None):
    plt.figure()

    idx = np.argsort(x.flatten())
    x_sorted = x[idx]
    y_sorted = y[idx]

    plt.plot(x_sorted, y_sorted, marker='o', color='blue', label="Past Data")

    if pred_x is not None:
        plt.scatter(pred_x, pred_y, color='red', s=120, label="Predicted Value")

    plt.title("Smart Grid Load Prediction")
    plt.xlabel("Day / Hour")
    plt.ylabel("Electricity Load")

    plt.legend()
    plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    graph = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return graph

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('dashboard'))
    return render_template("login.html")

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        return redirect(url_for('login'))
    return render_template("signup.html")

@app.route('/dashboard', methods=['GET','POST'])
def dashboard():
    prediction = None

    if request.method == 'POST' and model is not None:
        value = int(request.form['hour'])

        idx = (np.abs(X.flatten() - value)).argmin()
        line_value = y[idx]

        prediction = line_value
        graph = create_graph(X, y, [[value]], [line_value])

    else:
        graph = create_graph(X, y)

    return render_template(
        "index.html",
        prediction=prediction,
        graph=graph,
        accuracy=accuracy
    )

if __name__ == "__main__":
    app.run(debug=True)

app = app
