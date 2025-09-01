# Minimal TensorFlow fine-tuning stub (classification-style; adapt to your data/task)
import tensorflow as tf
import numpy as np

def train_demo():
    X = np.random.randn(256, 300).astype('float32')
    y = (X.mean(axis=1)>0).astype('int32')
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(300,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)
    model.save('configs/tf_demo.keras')

if __name__ == '__main__':
    train_demo()
