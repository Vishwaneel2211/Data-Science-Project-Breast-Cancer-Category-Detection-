from flask import Flask, render_template, request
import joblib
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

# Load models
extra_trees = joblib.load("extra_trees.pkl")
random_forest = joblib.load("random_forest.pkl")

# Load XGBoost using native format
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgboost.json")

# Features used by the model
features = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean"
]


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    benign_probability = None
    malignant_probability = None

    if request.method == "POST":

        values = []

        for feature in features:
            value = float(request.form[feature])
            values.append(value)

        input_data = pd.DataFrame(
            [values],
            columns=features
        )

        # Get probabilities from all three models
        et_proba = extra_trees.predict_proba(input_data)[0]
        rf_proba = random_forest.predict_proba(input_data)[0]
        xgb_proba = xgb_model.predict_proba(input_data)[0]

        # Soft voting: average probabilities
        final_proba = (
            et_proba +
            rf_proba +
            xgb_proba
        ) / 3

        prediction = float(final_proba.argmax())

        benign_probability = round(
            final_proba[0] * 100, 2
        )

        malignant_probability = round(
            final_proba[1] * 100, 2
        )

    return render_template(
        "index.html",
        prediction=prediction,
        benign_probability=benign_probability,
        malignant_probability=malignant_probability
    )


if __name__ == "__main__":
    app.run(debug=True)