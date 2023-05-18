import webbrowser

from waitress import serve
from flask import Flask, request, render_template
from scripts.model import *
from threading import Timer


# Create flask app
app = Flask(__name__, template_folder='templates')


def open_browser():
    webbrowser.open_new('http://127.0.0.1:8080')


@app.route('/', methods=['GET', "POST"])
@app.route('/home', methods=['GET', "POST"])
def home():
    if request.method == "POST":
        sentiment = request.form.get("sentiment")
        selected_model = request.form.get("model").lower()
        selected_cluster = request.form.get("cluster")
        text = 'Your sentiment is: ' + sentiment

        if sentiment == '' or selected_model == '' or selected_cluster == '':
            output = 'Invalid input'

        else:
            sentiment = DataFrame([sentiment], columns={'text'})
            sentiment_vectorized = preprocess(sentiment=sentiment)

            if len(sentiment_vectorized) != 0:
                if selected_cluster == "auto":
                    selected_cluster = cluster(sentiment=sentiment_vectorized)
                prediction = classify(sentiment=sentiment_vectorized, model=selected_model, cluster=selected_cluster)
                output = prediction
                
            else:
                output = 'Invalid input or your sentiment contains unrecognized text'

    else:
        output = ''
        text = ''

    return render_template('index.html', output=output, text=text)


if __name__ == '__main__':
    Timer(1, open_browser).start()
    serve(app, host='0.0.0.0', port=8080)