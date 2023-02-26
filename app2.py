
from flask import Flask, request, render_template
import pickle

LR = pickle.load(open('final_model.pickle', 'rb'))
vect = pickle.load(open('vect.pickle', 'rb'))

#LR = pickle.load(open('model.pickle', 'rb'))
#vect = pickle.load(open('vector.pickle', 'rb'))

app = Flask(__name__)

@app.route('/')
def issue():
    return render_template('index.html')

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        text = request.form['nm']
        result_pred = LR.predict(vect.transform([text]))
        return render_template("result.html", result = result_pred.flatten()[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080) 