import sys
from pathlib import Path

# Add parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
from flask import Flask, request, jsonify
from data_pipeline.prediction_pipeline import prediction_pipeline

app = Flask(__name__)


@app.route("/")
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Sentiment Analysis</title>
    <style>
        /* CSS for centering the form */
        body, html {
            height: 100%;
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            font-family: Arial, sans-serif;
        }

        .container {
            width: 400px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #f9f9f9;
        }
        [type="checkbox"]
                    {
            vertical-align:middle;
            
                    }
        textarea, select {
            margin-bottom: 20px;
            width: 100%;
        }
        #predictionResult {
        font-size: 16px;
        font-weight: bold;
    }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sentiment Analysis Prediction</h1>
        <form id = "FormId">
            <label for="text">Enter text:</label><br>
            <textarea id="text" name="text" rows="4" cols="50"></textarea><br>
            <label for="model_type">Choose a model:</label>&nbsp;
            <select id="model_type" name="model_type">
                <option value="nb">Naive Bayes</option>
                <option value="nn">Neural Network</option>
            </select><br>
            <label  style="word-wrap:break-word" for="synthetic">Use synthetic data: 
            <input  type="checkbox" id="synthetic" name="synthetic">
            </label><br>
            <div style = "margin-top: 20px" "margin-button: 20px">
            <input type="submit" value="Submit">
            </div>
        </form>
        <div id="predictionResult"></div>

        <div id = "predictionText">
        <label for="ptext">Input/Generated Text:</label><br>
        <textarea id="ptext" readonly = "readonly" rows="4" cols="50"></textarea><br>
        </div>
    </div>
    
    <script>
        // Function to set the color of the prediction result
        function setPredictionResultColor(prediction) {
            var resultElement = document.getElementById("predictionResult");
            if (prediction.toLowerCase() === "positive") {
                resultElement.style.color = 'green';
            } else if (prediction.toLowerCase() === "negative") {
                resultElement.style.color = 'red';
            } else {
                resultElement.style.color = 'black'; // Default color for neutral or unknown predictions
            }
        }

        document.getElementById("FormId").addEventListener("submit", function(e) {
            e.preventDefault();

            var formData = {
                text: document.getElementById("text").value,
                model_type: document.getElementById("model_type").value,
                synthetic: document.getElementById("synthetic").checked
            };

            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            })
            .then(response => {
                if (response.headers.get("content-type").includes("application/json")) {
                    return response.json();
                }
                throw new TypeError("Oops, we haven't got JSON!");
            })
            .then(data => {
                var resultElement = document.getElementById("predictionResult");
                var ptextElement = document.getElementById("ptext");
                if (data.error) {
                    resultElement.innerHTML = `<p>Error: ${data.error}</p>`;
                    resultElement.style.color = 'black'; // Default color for error messages
                } else {
                    resultElement.innerHTML = `<p>Prediction: ${data.prediction}</p>`;
                    ptextElement.value = `${data.text}`;
                    setPredictionResultColor(data.prediction); // Set the color based on the prediction
                }
            })
            .catch(error => {
                console.error('Error:', error);
                var resultElement = document.getElementById("predictionResult");
                resultElement.innerHTML = `<p>An error occurred: ${error}</p>`;
                resultElement.style.color = 'black'; // Default color for error messages
            });
        });
    </script>
</body>
</html>"""


@app.route("/predict", methods=["POST"])
def predict():
    if request.content_type != "application/json":
        return jsonify({"error": "Content-Type not supported"}), 415

    data = request.json

    # Basic input validation
    if "text" not in data or "model_type" not in data or "synthetic" not in data:
        return jsonify({"error": "Missing required parameters"}), 400

    text = data["text"]
    model_type = data["model_type"]
    synthetic = data["synthetic"]

    if (not isinstance(text, str) or text.strip() == "") and not synthetic:
        return jsonify({"error": "Text must be a non-empty string"}), 400

    if model_type not in ["nb", "nn"]:
        return jsonify({"error": "Invalid model type"}), 400

    if not isinstance(synthetic, bool):
        return jsonify({"error": "Synthetic must be a boolean"}), 400

    # Prediction
    text, prediction = prediction_pipeline(text, model_type, synthetic)

    # Convert NumPy array to list if it's an array
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    return (
        jsonify({"text": text, "prediction": prediction}),
        200,
        {"ContentType": "application/json"},
    )


if __name__ == "__main__":
    app.run(debug=True)
