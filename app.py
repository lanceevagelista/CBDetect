from flask import Flask, render_template, request
import requests
from cyberbullying_detection import load_all_hate_words, detect_cyberbullying

app = Flask(__name__)

apify_token = 'apify_api_tUDb9D42dd3hL2Q2DAD0SZqlzQclD21EEZd1'
url = 'https://api.apify.com/v2/acts/apify~instagram-comment-scraper/run-sync-get-dataset-items?token=' + apify_token

hate_words_file_paths = ['profanity_en.csv', 'profanity_fil.csv', 'profanity_es.csv']
hate_words = load_all_hate_words(hate_words_file_paths)

def contains_profanity(text):
    # Check if any profane word is present in the text
    return any(word in text.lower() for word in hate_words)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    comments = []

    if request.method == 'POST':
        text_input = request.form.get('text_input', '')  # Use .get() to avoid KeyError
        if text_input:
            # Define input data
            data = {
                "directUrls": [text_input]
            }

            response = requests.post(url, json=data)

            # Check for cyberbullying in any comment
            detected_cyberbullying = False
            for items in response.json():
                comment_text = items.get('text', '').lower()  # Convert to lowercase
                comments.append(comment_text)

                # Check for profanity
                if contains_profanity(comment_text):
                    # Check for cyberbullying
                    cyberbullying_result, _ = detect_cyberbullying(comment_text, hate_words)
                    if cyberbullying_result == 'Cyberbullying Detected':
                        detected_cyberbullying = True

            # Set result based on cyberbullying detection
            if detected_cyberbullying:
                result = "Cyberbullying Detected"
            else:
                result = 'No Cyberbullying Detected'

    return render_template('index.html', result=result, comments=comments)



if __name__ == '__main__':
    app.run(debug=True)