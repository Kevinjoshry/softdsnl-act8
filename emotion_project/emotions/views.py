from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt 
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import json

# --- 1. Define the Emotion Label Mapping (28 Classes) ---
# This list maps the integer index (0-27) to the emotion name string.
GO_EMOTIONS_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'contempt', 'desire', 'disappointment', 'disapproval', 
    'disgust', 'embarrassment', 'excitement', 'fear', 'grief', 
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
    'relief', 'remorse', 'sadness', 'surprise', 'neutral', 'unannotated' # Index 27 is the unannotated class
]

# --- 2. Global Model Loading ---
# Load once when the server starts
MAX_LEN = 50 # Must match the MAX_LEN used in training (train_emotions.py)

try:
    # Ensure these files (emotion_model.h5, tokenizer.pkl, encoder.pkl) are 
    # accessible from the directory where Django is run (usually the project root).
    model = tf.keras.models.load_model("emotion_model.h5")
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
    print("Model, tokenizer, and encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model assets: {e}")
    model, tokenizer, encoder = None, None, None
# -------------------------------------------------------------

@csrf_exempt
def predict_emotion(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST requests are accepted"}, status=405)

    if not model or not tokenizer or not encoder:
        return JsonResponse({"error": "Model assets not loaded"}, status=500)

    try:
        # Load JSON body
        body = json.loads(request.body.decode('utf-8'))
        text = body.get("text", "")
        
        if not text:
            return JsonResponse({"error": "Missing 'text' field in request body"}, status=400)

        # Preprocess the text
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # Predict
        prediction = model.predict(padded)
        
        # Get the index of the highest probability (e.g., 16 for 'joy', or 27 for 'unannotated')
        predicted_index = np.argmax(prediction, axis=1)[0]
        
        # We don't need encoder.inverse_transform if we only use the integer index, 
        # as the encoder was only fitted on range(28) anyway.
        
        # --- 3. Index to String Conversion ---
        # Look up the string name using the predicted index from our list
        emotion_name = GO_EMOTIONS_LABELS[predicted_index]
        
        # Final answer format now returns the string name
        return JsonResponse({"text": text, "predicted_emotion": emotion_name}) 

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)
    except IndexError:
        return JsonResponse({"error": "Prediction index out of bounds. Check model output size."}, status=500)
    except Exception as e:
        return JsonResponse({"error": f"An internal error occurred: {e}"}, status=500)