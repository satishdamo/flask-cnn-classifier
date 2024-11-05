import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI
from playsound import playsound
import tempfile

load_dotenv()  # take environment variables from .env

client = OpenAI(
    # Defaults to os.environ.get("OPENAI_API_KEY")
)

def convert_text_to_speech(input_text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=f'Hi, your input is {input_text}'
    )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file: 
        temp_file_path = temp_file.name 
    
    response.stream_to_file(temp_file_path) 
    #print(f"File saved at {temp_file_path}")
    #playsound(temp_file_path)

app = Flask(__name__)
model = load_model('cifar10_cnn_model.h5')
lables=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def show_class_name(label_index):
    return lables[label_index].capitalize()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file.stream)
    processed_image = preprocess_image(image, target_size=(32, 32))
    
    predictions = model.predict(processed_image)
    label = np.argmax(predictions[0])
    
    #convert lable to speech and play the speech
    convert_text_to_speech(show_class_name(int(label)))
    
    return jsonify({'label': show_class_name(int(label))})

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=5000)
