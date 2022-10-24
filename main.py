import numpy as np
import streamlit as st
import os
import cv2 as cv2
import tensorflow as tf
keras = tf.keras
from keras import layers
from keras.models import Sequential


@st.cache
def load_data(class_names_label, IMAGE_SIZE):
    DIRECTORY = 'data'
    CATEGORY = ['train', 'test']

    output = []

    for category in CATEGORY:
        path = os.path.join(DIRECTORY, category)
        images = []
        labels = []

        for folder in os.listdir(path):
            label = class_names_label[folder]

            for file in os.listdir(os.path.join(path, folder)):
                img_path = os.path.join(os.path.join(path, folder), file)

                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                images.append(image)
                labels.append(label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')

        output.append((images, labels))

    return output


@st.cache
def normalize(train_images, test_images):
    train_images = np.array(train_images) / 255
    test_images = np.array(test_images) / 255

    return train_images, test_images


@st.cache(allow_output_mutation=True)
def build_model(train_images, train_labels, test_images, test_labels, epoch):
    model = Sequential([
        layers.Input(shape=train_images.shape[1:]),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, validation_split=0.2, epochs=epoch)

    accuracy = model.evaluate(test_images, test_labels, verbose=2)[1]

    return model, accuracy


def load_predict_image(model, uploaded_file, IMAGE_SIZE):
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)

        st.image(image)

        img_batch = np.expand_dims(image, axis=0)

        return model.predict(img_batch)


def main():
    # Page config
    st.set_page_config(page_title='Pneumonia Detection')
    st.title('Pneumonia Detection')
    st.subheader('Load a chest xray image and the algorithm will do the rest.')

    # Var definition
    class_names = ['PNEUMONIA', 'NORMAL']
    class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
    IMAGE_SIZE = (180, 180)

    # Load data
    (train_images, train_labels), (test_images, test_labels) = load_data(class_names_label, IMAGE_SIZE)

    # Normalize data
    train_images, test_images = normalize(train_images, test_images)

    # Build model & get accuracy
    mdl, acc = build_model(train_images, train_labels, test_images, test_labels, 2)

    # Load & predict image
    uploaded_file = st.file_uploader("Choose an image (JPEG format only)", 'jpeg')

    if uploaded_file is not None:
        prediction = load_predict_image(mdl, uploaded_file, IMAGE_SIZE)

    # Display results
        st.subheader(f"Prediction: {class_names[np.argmax(prediction[0])]}")
        st.subheader(f"Model accuracy: {'{:.0%}'.format(acc)}")


if __name__ == "__main__":
    main()
