import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("deepfake_model.keras")

img_path = "test.jpg"   # test image name

img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print("FAKE IMAGE")
else:
    print("REAL IMAGE")