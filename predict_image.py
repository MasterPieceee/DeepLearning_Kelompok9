import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

model = tf.keras.models.load_model("mnist_model_local.keras")

image_path = "angka9.png"   # ganti sesuai nama file kamu

img = Image.open(image_path).convert("L")  # grayscale
img = img.resize((28, 28))

img_array = np.array(img)

img_array = img_array / 255.0

img_input = img_array.reshape(1, 28, 28)

pred = model.predict(img_input)
hasil = np.argmax(pred)

print("Hasil prediksi:", hasil)
print("Probabilitas:", pred)

plt.imshow(img_array, cmap="gray")
plt.title(f"Prediksi: {hasil}")
plt.axis("off")
plt.show()