from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the model
model = load_model("D:\\stresswebsite\\NSP.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            X = float(request.form['X'])
            Y = float(request.form['Y'])
            Z = float(request.form['Z'])
            EDA = float(request.form['EDA'])
            HR = float(request.form['HR'])
            TEMP = float(request.form['TEMP'])

            # Make prediction
            inputs = [X, Y, Z, EDA, HR, TEMP]
            result = model.predict(np.array([inputs]))
            predicted_class = np.argmax(result)

            # Customizing the response to display the result
            if predicted_class == 0:
                result_text = "Low Stress"
                solution = "You're doing great! Maintain your healthy lifestyle."
                music_link = None
            elif predicted_class == 1:
                result_text = "Moderate Stress"
                solution = "Try some relaxation techniques like yoga or listening to music."
                music_link = "https://open.spotify.com/playlist/<your_playlist_id>"  # Spotify playlist link
            elif predicted_class == 2:
                result_text = "High Stress"
                solution = "It's important to manage your stress. Consider activities like meditation or going for a walk."
                music_link = "https://open.spotify.com/playlist/<your_playlist_id>"  # Spotify playlist link
            else:
                result_text = "Unknown Stress Level"
                solution = "Please try again later."
                music_link = None

            return render_template('result.html', result=result_text, solution=solution, music_link=music_link)
        except Exception as e:
            # Handle any exceptions
            return render_template('error.html', message="Error: {}".format(str(e)))

if __name__ == '__main__':
    app.run(debug=True)
