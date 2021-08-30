# https://github.com/samuelvinicius777/classificador_dogs_cats.git
# Carregandos imports

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, save_model

# Defindo a escala de cor dos pixels

train = ImageDataGenerator(rescale= 1./255)
validation = ImageDataGenerator(rescale= 1./255)
test = ImageDataGenerator(rescale=1./255)


# Carregamento do dataset train/validation/test

train_dataset = train.flow_from_directory('data/training_set_dogs_cats/training_set',
                                          target_size= (280,280),
                                          color_mode = "rgb",
                                          batch_size = 32,
                                          class_mode = 'categorical')

validation_dataset = validation.flow_from_directory('data/test_set_dogs_cats/test_set_validation',
                                          target_size= (280,280),
                                          color_mode = "rgb",          
                                          batch_size = 32,
                                          class_mode = 'categorical')

test_dataset = test.flow_from_directory('data/test_set_dogs_cats/test_set',
                                          target_size= (280,280),
                                          color_mode = "rgb",
                                          batch_size = 32,
                                          class_mode = 'categorical')


# Construção do modelo

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32,(3,3), activation= "relu", input_shape = (280,280,3)),
                                    tf.keras.layers.Conv2D(64,(3,3), activation= "relu"),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64,activation= "relu"),                                  
                                    tf.keras.layers.Dense(2,activation='softmax')
                                    ])

# Compilação do modelo

model.compile(loss= "categorical_crossentropy",
              optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics = ["accuracy"])

# Treinamento do modelo

model_fit = model.fit(train_dataset,
                      steps_per_epoch = 30,
                      batch_size = 32,
                      epochs = 50,
                      verbose = 1,
                      validation_data = validation_dataset)

# Salvando o modelo

model.save("models/model.h5")
