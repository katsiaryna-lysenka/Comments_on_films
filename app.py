from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from flask import Flask, request, render_template, redirect
import tensorflow as tf
from tensorflow.keras.saving import save_model

app = Flask(__name__)

# Загрузка данных
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Предобработка данных
maxlen = 256
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')

# Создание модели
model = Sequential([
    Embedding(10000, 64, trainable=False),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Компиляция модели с оптимизатором Adam и установкой learning rate
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Инициализация и обучение Tokenizer
tokenizer = Tokenizer()
texts = [' '.join([str(word) for word in seq]) for seq in x_train]
tokenizer.fit_on_texts(texts)

# Обучение модели
model.fit(x_train, y_train, epochs=5, batch_size=512, validation_data=(x_test, y_test))

# Сохранение модели
save_model(model, 'my_model_updated_lr.keras')

# Загрузка модели
model = tf.keras.models.load_model('my_model_updated_lr.keras')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        new_review_sequences = tokenizer.texts_to_sequences([review])
        new_review_padded = pad_sequences(new_review_sequences, value=0, padding='post', maxlen=256)
        prediction = model.predict(new_review_padded)[0][0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        return render_template('index.html', prediction_text=f'The sentiment is {sentiment}')

@app.route('/new_comment')
def new_comment():
    return redirect("http://127.0.0.1:5000/")

if __name__ == '__main__':
    app.run(debug=True)
