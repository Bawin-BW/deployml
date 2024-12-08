# prompt: buatkan flask API beserta tokenizernya

from flask import Flask, request, jsonify
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the model and tokenizer
model = load_model('sentiment_analysis_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

vocab_size = 1000
oov_tok = "<OOV>"
padding_type = "post"
max_length = 100

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text']
        
        if not isinstance(text, list):
            text = [text]  # Ensure text is always a list

        inference_sequences = tokenizer.texts_to_sequences(text)
        inference_padded = pad_sequences(inference_sequences, padding=padding_type, maxlen=max_length)
        result = model.predict(inference_padded)
        
        predictions = []
        for res in result:
          predictions.append("positif" if res[0] > 0.5 else "negatif")

        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)