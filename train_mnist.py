import tensorflow as tf
from tensorflow.keras import layers, models


# 1. Load dataset MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalisasi
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Bangun model
model = models.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 4. Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train model (tanpa EarlyStopping, tanpa callback sama sekali)
print("Mulai training...")
model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# 6. Evaluasi di test set
print("\nEvaluasi di test set:")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_acc)

# 7. Simpan model
model.save("mnist_model_local.keras")
print("Model saved to mnist_model_local.keras")