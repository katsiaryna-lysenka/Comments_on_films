from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras.saving import save_model


def train_model():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

    maxlen = 256
    x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
    x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')

    model = Sequential([
        Embedding(10000, 64, trainable=False),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    tokenizer = Tokenizer()
    texts = [' '.join([str(word) for word in seq]) for seq in x_train]
    tokenizer.fit_on_texts(texts)

    model.fit(x_train, y_train, epochs=5, batch_size=512, validation_data=(x_test, y_test))

    save_model(model, 'my_model_updated_lr.keras')


if __name__ == "__main__":
    train_model()
