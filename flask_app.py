from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import os
import sys

flask_app = Flask(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Load model and tokenizer
model = load_model('/home/katsiaryna/PycharmProjects/app/my_model_updated_lr.keras')
tokenizer = Tokenizer()


@flask_app.route('/')
def home():
    return render_template('index.html')


@flask_app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        new_review_sequences = tokenizer.texts_to_sequences([review])
        new_review_padded = pad_sequences(new_review_sequences, value=0, padding='post', maxlen=256)
        prediction = model.predict(new_review_padded)[0][0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        return render_template('index.html', prediction_text=f'The sentiment is {sentiment}')


@flask_app.route('/new_comment')
def new_comment():
    return redirect("http://127.0.0.1:5000/")


if __name__ == '__main__':
    flask_app.run(debug=True)




