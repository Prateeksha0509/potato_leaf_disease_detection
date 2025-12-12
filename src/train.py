from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf

IMAGE_SIZE = (224,224)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)

train = datagen.flow_from_directory("dataset", target_size=IMAGE_SIZE, batch_size=32, subset="training")
val = datagen.flow_from_directory("dataset", target_size=IMAGE_SIZE, batch_size=32, subset="validation")

history = model.fit(train, validation_data=val, epochs=12)
model.save("models/potato_model.h5")