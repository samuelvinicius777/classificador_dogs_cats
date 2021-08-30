import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

classifier = load_model("models/model3.h5")

classifier.summary()

def load_image(img_path, show='true'):
    img_original = image.load_img(img_path)
    img = image.load_img(img_path, target_size=(280,280))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0
    #img_tensor = np.vstack([img_tensor])
    if show: 
        plt.xticks(range(10))
        plt.imshow(img_original)
        plt.axis('off')
        #plot_value_array(img_path, predictions)
        plt.show()
    return img_tensor


dir_path = 'data/test_set_dogs_cats/test_set/dogs'
count1 = 0
count2 = 0
for i in os.listdir(dir_path):
    new_image = load_image(dir_path+'//'+i)
    pred = classifier.predict(new_image)
    if np.argmax(pred) == 0:
        count1 = count1 + 1
        print("CAT")
    else:
        count2 = count2 + 1
        print("DOG")

    print(np.argmax(pred))
print(count1, "CAT")
print(count2, "DOG")