from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
import re

app = Flask(__name__)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "laptop_price.csv")

# Check if files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"model.pkl not found at {MODEL_PATH}")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"laptop_price.csv not found at {DATA_PATH}")

model = pickle.load(open(MODEL_PATH, "rb"))

# Load dataset (for column reference)
df = pd.read_csv(DATA_PATH)
X = df.drop("Price_euros", axis=1)
X = pd.get_dummies(X, drop_first=True)

def extract_weight_kg(weight_str):
    """Extract numeric weight from strings like '1.5kg' or '2.2 kg'"""
    if isinstance(weight_str, (int, float)):
        return float(weight_str)
    match = re.search(r'(\d+\.?\d*)', str(weight_str))
    return float(match.group(1)) if match else 0.0

def extract_memory_gb(memory_str):
    """Extract total memory in GB from strings like '256GB SSD', '1TB HDD'"""
    if isinstance(memory_str, (int, float)):
        return float(memory_str)
    
    memory_str = str(memory_str).upper()
    total_gb = 0
    
    # Check for TB
    tb_match = re.search(r'(\d+\.?\d*)\s*TB', memory_str)
    if tb_match:
        total_gb += float(tb_match.group(1)) * 1024
    
    # Check for GB
    gb_match = re.search(r'(\d+\.?\d*)\s*GB', memory_str)
    if gb_match:
        total_gb += float(gb_match.group(1))
    
    return total_gb if total_gb > 0 else 0.0

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Extract and clean weight
            weight_raw = request.form.get('weight', '0')
            weight_kg = extract_weight_kg(weight_raw)
            
            # Extract and clean memory
            memory_raw = request.form.get('memory', '')
            memory_gb = extract_memory_gb(memory_raw)
            
            new_laptop = {
                'Inches': float(request.form.get('inches', 0)),
                'Ram': int(request.form.get('ram', 0)),
                'Memory': memory_raw,  # Keep original for one-hot encoding
                'Company': request.form.get('company', ''),
                'TypeName': request.form.get('type', ''),
                'ScreenResolution': request.form.get('resolution', ''),
                'Cpu': request.form.get('cpu', ''),
                'Gpu': request.form.get('gpu', ''),
                'OpSys': request.form.get('os', ''),
                'Weight': weight_kg  # Use numeric weight
            }
            
            # Validate required fields
            if not new_laptop['Company'] or not new_laptop['Cpu']:
                error = "Company and CPU are required fields"
            else:
                new_df = pd.DataFrame([new_laptop])
                new_df = pd.get_dummies(new_df)
                new_df = new_df.reindex(columns=X.columns, fill_value=0)
                prediction = round(model.predict(new_df)[0], 2)
                
        except Exception as e:
            error = f"Error making prediction: {str(e)}"

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
