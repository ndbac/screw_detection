import tensorflow as tf
from tensorflow.keras import layers, models
import os

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Model parameters
IMG_HEIGHT = 626
IMG_WIDTH = 285
BATCH_SIZE = 32
EPOCHS = 10

def load_and_preprocess_data(data_dir):
    # Load images from directory with labels based on folder names
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    
    return dataset, val_dataset

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def train_model(model, train_dataset, val_dataset):
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS
    )
    
    return history

def predict_image(model, image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    prediction = model.predict(img_array)
    return "Screw Present" if prediction[0] > 0.5 else "Screw Missing"

if __name__ == "__main__":
    # data/screw_present/ and data/screw_missing/
    data_dir = "data"

    # Load and preprocess data
    train_dataset, val_dataset = load_and_preprocess_data(data_dir)

    # Create and train model
    model = create_model()
    history = train_model(model, train_dataset, val_dataset)

    # Save the model
    model.save('screw_detection_model.h5')

    # Example prediction
    test_image_path = "test_image.jpg"
    if os.path.exists(test_image_path):
        result = predict_image(model, test_image_path)
        print(f"Prediction: {result}")
