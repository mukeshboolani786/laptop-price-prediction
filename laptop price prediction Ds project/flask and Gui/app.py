from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))

# Load dataset (for column reference)
df = pd.read_csv(os.path.join(BASE_DIR, "laptop_price.csv"))
X = df.drop("Price_euros", axis=1)
X = pd.get_dummies(X, drop_first=True)

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None

    if request.method == "POST":

        new_laptop = {
            'Inches': float(request.form['inches']),
            'Ram': int(request.form['ram']),
            'Memory': request.form['memory'],
            'Company': request.form['company'],
            'TypeName': request.form['type'],
            'ScreenResolution': request.form['resolution'],
            'Cpu': request.form['cpu'],
            'Gpu': request.form['gpu'],
            'OpSys': request.form['os'],
            'Weight': request.form['weight']
        }

        new_df = pd.DataFrame([new_laptop])
        new_df = pd.get_dummies(new_df)
        new_df = new_df.reindex(columns=X.columns, fill_value=0)

        prediction = model.predict(new_df)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
