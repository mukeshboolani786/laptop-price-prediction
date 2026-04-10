from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    # 🔹 CSV ka correct path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "laptop_price.csv")

    # 🔹 CSV load
    df = pd.read_csv(csv_path)


    tables = []

    if request.method == "POST":
        company = request.form.get("company")

        if company:
            result = df[df["Company"].str.contains(company, case=False, na=False)]
            tables = [result.to_html(classes="table table-striped", index=False)]

    return render_template("index.html", tables=tables)

if __name__ == "__main__":
    app.run(debug=True)








