# Uncomment this, if you are getting annoying tensorflow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from create_classifier import create_model

# TO TRAIN MODEL FROM ZERO: uncomment the line below
# model = create_model()

# TO LOAD THE PRE-TRAINED MODEL: uncomment the line below
model = load_model("model.h5")
path = "single_prediction/cat_or_dog_4.jpg"  # Put here the path to image
test_image = image.load_img(path, target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

if result == 1:
    print("DOG")
else:
    print("CAT")
