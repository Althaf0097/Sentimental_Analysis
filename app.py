from flask import Flask, render_template, request, jsonify
import pickle
import logging
import numpy as np  # Add numpy for type conversion

# Load the models and vectorizers for each language
models = {
    'english': {
        'model': pickle.load(open('./english_lr.pkl', 'rb')),
        'vectorizer': pickle.load(open('./english_tfidf_vectorizer.pkl', 'rb'))
    },
    'hindi': {
        'model': pickle.load(open('./hindi_model.pkl', 'rb')),
        'vectorizer': pickle.load(open('./hindi_tfidf_vectorizer.pkl', 'rb'))
    }
}

# Ensure sentiment mapping matches the models' outputs
sentiment_mapping = {
    0: 'Negative',
    1: 'Neutral',
    2: 'Positive'
}

# Create Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Set up logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def home():
    """Render the home page."""
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle sentiment prediction requests."""
    try:
        review = request.form.get('review')  # Get review from form data
        language = request.form.get('language')  # Get selected language from form data

        # Handle empty or invalid inputs
        if not review or not review.strip():
            return jsonify({'error': 'Empty review. Please enter some text.'})

        if language not in models:
            return jsonify({'error': 'Invalid language selected.'})

        # Preprocess the review
        preprocessed_review = review.lower().strip()  # Lowercase for consistency

        # Log preprocessed review
        logging.info(f"Preprocessed review: {preprocessed_review}")

        # Load the corresponding model and vectorizer
        model = models[language]['model']
        vectorizer = models[language]['vectorizer']

        # Vectorize the input text using the corresponding vectorizer
        try:
            review_vectorized = vectorizer.transform([preprocessed_review])
        except Exception as e:
            logging.error(f"Error during vectorization: {str(e)}")
            return jsonify({'error': f'Error in vectorization: {str(e)}'})

        # Predict sentiment and probabilities
        try:
            prediction = model.predict(review_vectorized)[0]  # Get the predicted class
            probabilities = model.predict_proba(review_vectorized)[0]  # Get the probabilities for all classes
            sentiment = sentiment_mapping.get(prediction, 'Unknown')  # Map the predicted class to sentiment
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': f'Error in prediction: {str(e)}'})

        # Convert probabilities from np.float32 to native Python float
        probability_dict = {
            'Negative': round(float(probabilities[0]) * 100, 2),
            'Neutral': round(float(probabilities[1]) * 100, 2) if len(probabilities) > 2 else None,
            'Positive': round(float(probabilities[-1]) * 100, 2)
        }

        # Log the result
        logging.info(f"Prediction: {sentiment}, Probabilities: {probability_dict}")

        return jsonify({
            'sentiment': sentiment,
            'probabilities': probability_dict
        })

    except Exception as e:
        logging.error(f"General error in prediction: {str(e)}")
        return jsonify({'error': f'Error in prediction: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=True)
