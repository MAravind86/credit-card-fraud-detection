## FraudGuard — Credit Card Fraud Detection (Flask + ML)

A Flask web app and REST API for detecting fraudulent credit card transactions. It trains a RandomForest model on the well-known credit card fraud dataset, serves predictions via an API, and provides a modern Tailwind UI to try transactions interactively.

### Features
- Real-time prediction API (`/api/predict`)
- Health check (`/api/health`)
- Modern dashboard UI at `/` with live detection form
- Automatic data loading, balancing, training, and model persistence (`fraud_model.pkl`, `fraud_scaler.pkl`)

### Project Structure
```
app.py                     # Flask app (routes: UI, /api/predict, /api/health)
model_service.py           # Thread-safe singleton service around the ML detector
fraud_detector_interface.py# SimpleFraudDetector: data prep, training, predict
templates/index.html       # Tailwind UI dashboard and live detection form
requirements.txt           # Python dependencies
creditcard.csv (or .zip)   # Dataset (Kaggle credit card fraud)
fraud_model.pkl            # Saved RandomForest model (generated)
fraud_scaler.pkl           # Saved RobustScaler (generated)
```

### Prerequisites
- Python 3.9+ recommended
- pip

### Installation
PowerShell (Windows):
```powershell
cd "C:\Users\M.Aravind\Desktop\New folder (3)"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you only have `creditcard.zip`, unzip it to create `creditcard.csv` in the project root.

### Running the App
```powershell
python app.py
```
The server starts at `http://127.0.0.1:8000`.

On first run, the service will:
1) Load `creditcard.csv`
2) Engineer features and balance classes
3) Train the RandomForest model
4) Save `fraud_model.pkl` and `fraud_scaler.pkl`
5) Load the saved model for serving

Subsequent runs will load the existing `.pkl` files if present.

### Using the Web UI
Open `http://127.0.0.1:8000` in your browser.
- Go to the “Live Detection” tab
- Enter Transaction ID, Amount (₹), and Time (`now`, `HH:MM`, or epoch seconds)
- Submit to see prediction, fraud probability, risk level, and risk factors

### API
Base URL: `http://127.0.0.1:8000`

- Health
  - GET `/api/health`
  - Response: `{ status, model_loaded, message }`

- Predict
  - POST `/api/predict`
  - JSON body:
    ```json
    {
      "transaction_id": "TXN12345",
      "time": "now"  // or "14:30" or 1737480000,
      "amount": 499.99
    }
    ```
  - Example (PowerShell curl):
    ```powershell
    curl -Method POST \
      -Uri http://127.0.0.1:8000/api/predict \
      -Headers @{ 'Content-Type' = 'application/json' } \
      -Body '{"transaction_id":"TXN1","time":"now","amount":799.5}'
    ```
  - Example (curl):
    ```bash
    curl -X POST http://127.0.0.1:8000/api/predict \
      -H 'Content-Type: application/json' \
      -d '{"transaction_id":"TXN1","time":"14:30","amount":799.5}'
    ```

Response fields include: `transaction_id`, `timestamp`, `amount`, `fraud_prediction` (FRAUD/LEGITIMATE), `fraud_probability` (0-1), `risk_level` (LOW/MEDIUM/HIGH), `confidence`, and `risk_factors`.

### Dataset
This project expects the Kaggle credit card fraud dataset as `creditcard.csv` in the project root. If you have `creditcard.zip`, unzip it first. The training process may take a few minutes depending on your machine.

### Troubleshooting
- Missing dataset: Ensure `creditcard.csv` exists in the project root.
- First run is slow: Training happens on first startup; later runs load `.pkl` files.
- Port in use: Change the port in `app.py` (`app.run(..., port=8000, ...)`).
- Invalid input errors: `time` must be `now`, `HH:MM`, or epoch seconds; `amount` must be numeric.

### License
For educational/demo purposes. Replace with your preferred license.


