from django.shortcuts import render
from pathlib import Path
import pickle
import numpy as np

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model path
model_path = BASE_DIR / "predictor" / "models" / "model_bundle.pkl"

# Global variables
model = None
commodity_encoder = None
state_encoder = None
market_encoder = None


# ✅ Load model safely
def load_model():
    global model, commodity_encoder, state_encoder, market_encoder

    if model is None:
        with open(model_path, "rb") as f:
            data = pickle.load(f)

            model = data["model"]
            commodity_encoder = data["commodity_encoder"]
            state_encoder = data["state_encoder"]
            market_encoder = data["market_encoder"]


# ✅ Home page
def home(request):
    load_model()

    return render(request, "index.html", {
        "crops": commodity_encoder.classes_,
        "states": state_encoder.classes_,
        "markets": market_encoder.classes_,
    })


# ✅ Prediction
def predict_price(request):
    load_model()

    if request.method == "POST":
        try:
            # Get string inputs
            crop = request.POST.get("crop")
            state = request.POST.get("state")
            market = request.POST.get("market")
            year = request.POST.get("year")
            month = request.POST.get("month")

            # Validate
            if not all([crop, state, market, year, month]):
                raise ValueError("All fields are required")

            # Convert numbers
            year = float(year)
            month = float(month)

            # Encode strings → numbers
            crop_encoded = commodity_encoder.transform([crop])[0]
            state_encoded = state_encoder.transform([state])[0]
            market_encoded = market_encoder.transform([market])[0]

            # Model input
            input_data = np.array([[crop_encoded, state_encoded, market_encoded, year, month]])

            # Predict
            prediction = model.predict(input_data)[0]

            result = f"Predicted Price: ₹{round(prediction, 2)}"

        except Exception as e:
            result = f"Error: {str(e)}"

        return render(request, "index.html", {
            "prediction_text": result,
            "crops": commodity_encoder.classes_,
            "states": state_encoder.classes_,
            "markets": market_encoder.classes_,
        })

    return home(request)