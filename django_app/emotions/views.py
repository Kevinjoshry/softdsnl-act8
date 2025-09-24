from django.http import JsonResponse
import tensorflow as tf
import numpy as np
import pickle
import json

model = tf.keras.models.load_model("emotion_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

def predict_emotion(request):
    if request.method == "POST":
        body = json.loads(request.body)
        text = body.get("text", "")

        seq = tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=50)
        prediction = model.predict(padded)
        emotion = encoder.inverse_transform([np.argmax(prediction)])[0]

        return JsonResponse({"text": text, "predicted_emotion": emotion})
