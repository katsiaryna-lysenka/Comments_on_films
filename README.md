# Sentiment Analysis App

This is a simple web application built with Flask for sentiment analysis. It predicts the sentiment (positive or negative) of a user-provided movie review.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/your-sentiment-analysis-app.git
   cd your-sentiment-analysis-app
   
## Create a virtual environment:
python -m venv venv

## Activate the virtual environment:

On Linux:
source venv/bin/activate

## Install dependencies:

pip install -r requirements.txt

## Run the train_model.py script:

python train_model.py

## Run the Flask application:

python flask_app.py

The app will be accessible at http://127.0.0.1:5000/.
Open your web browser and navigate to http://127.0.0.1:5000/.
You can enter a movie review in the provided form and click the "Predict" button to see the predicted sentiment.

## Testing
To run the unit tests, use the following command:

python -m unittest tests/test_app.py
python -m unittest tests/test_app_e2e.py
