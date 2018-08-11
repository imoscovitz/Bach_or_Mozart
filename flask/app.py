from flask import Flask, abort, render_template, jsonify, request
from api import make_prediction
#from werkzeug import secure_filename

import sys
sys.path.insert(0, '/Users/ilanmoscovitz/github/sf18_ds11/projects/03-mcnulty/')
import MTheory as mt

#UPLOAD_FOLDER = '/Users/ilanmoscovitz/github/sf18_ds11/projects/03-mcnulty/Music/Uploads'
#ALLOWED_EXTENSIONS = set(['mp3'])

app = Flask('MTheoryApp')
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

#@app.route('/music/<path:filename>')
#def download_file(filename):
#    return send_from_directory('/home/name/Music/', filename)

@app.route('/predict', methods=['POST'])
def my_form_post():
    audio_path = request.form['text']
    prediction = make_prediction(audio_path)

    return (render_template('winner.html', name=prediction['composer'], confidence=prediction['confidence']))

    #return jsonify(prediction)

#@app.route('/predict', methods=['POST'])
#def do_prediction():
#    if not request.json:
#        abort(400)
#    data = request.json
#
#    response = make_prediction(data)
#
#    return jsonify(response)
#/Users/ilanmoscovitz/github/sf18_ds11/projects/03-mcnulty/Music/Mozart/KV331_1_1_tema.mp3



app.run(debug=True)
