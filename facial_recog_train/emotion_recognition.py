import tensorflow as tf
import numpy as np
import cv2
import sys

# Load the trained model
model = tf.keras.models.load_model('updated_model.keras')

# Define the class names (adjust according to your dataset)
class_names = ['angry', 'contempt', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprise']

def preprocess_image(image_path):
    """
    Preprocess the input image to match the model's expected input format.
    :param image_path: Path to the input image.
    :return: Preprocessed image.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (48, 48))
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_emotion(image_path):
    """
    Predict the emotion of the person in the input image.
    :param image_path: Path to the input image.
    :return: Predicted emotion.
    """
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_emotion = class_names[predicted_class]
    return predicted_emotion

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python emotion_recognition.py <image_path>")
        sys.exit(1)

    #image_path = sys.argv[1]
    image_path = "teste.png"
    emotion = predict_emotion(image_path)
    print(f'The predicted emotion is: {emotion}')