from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.saving import save_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[df['label'].isin(['neg', 'pos'])]
    reviews = df['review'].astype(str).tolist()
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['label'])
    return reviews, labels


def train_model(file_path):
    reviews, labels = load_data(file_path)
    train_reviews, test_reviews, train_labels, test_labels = train_test_split(reviews, labels, test_size=0.2, random_state=42)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_reviews)
    train_sequences = tokenizer.texts_to_sequences(train_reviews)
    test_sequences = tokenizer.texts_to_sequences(test_reviews)
    maxlen = 256
    padded_train_sequences = pad_sequences(train_sequences, maxlen=maxlen, padding='post')
    padded_test_sequences = pad_sequences(test_sequences, maxlen=maxlen, padding='post')

    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, trainable=False),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(padded_train_sequences, train_labels, epochs=5, batch_size=512, validation_data=(padded_test_sequences, test_labels))

    model.save('my_model_updated_lr.keras')


if __name__ == "__main__":
    file_path = '/home/katsiaryna/Загрузки/katya_database.csv'
    train_model(file_path)
