from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

model = joblib.load('model/sentiment_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    tweet = None
    if request.method == 'POST':
        tweet = request.form['tweet']
        prediction = model.predict([tweet])[0]
        label_map = {1: 'Positive', 0: 'Negative', 2: 'Neutral'}
        sentiment = label_map.get(prediction, "Unknown")
    confusion_exists = os.path.exists('static/confusion_matrix.png')
    return render_template('index.html', sentiment=sentiment, tweet=tweet,
                           confusion_exists=confusion_exists)

if __name__ == '__main__':
    app.run(debug=True)