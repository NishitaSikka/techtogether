from flask import jsonify, Flask,render_template,url_for,request
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from preprocessing import *
import os
import pickle

model_dir = './models'
model = load_model(os.path.join(model_dir, 'weights_cpu.best.hdf5'))
    
with open(os.path.join(model_dir,'tokenizer.pickle'), 'rb') as handle:
    tokenizer = pickle.load(handle)
    
MAX_SEQUENCE_LENGTH = model.input_shape[1]
global graph
graph = tf.get_default_graph()

def rate_toxic(text):
    text_clean = clean_text(text)
    text_split = text_clean.split(' ')
    sequences = tokenizer.texts_to_sequences(text_split)
    sequences = [[item for sublist in sequences for item in sublist]]
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    with graph.as_default():
        predict = model.predict(data).reshape(-1,1)
    return predict
'''
toxic, severe_toxic, obsence, threat, insult, identity_hate = rate_toxic(text)
print('Prediction succesful!')
print(rate_toxic(text))
'''

app = Flask(__name__,static_url_path='/static')

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    if request.method == 'POST':
        txt_input = request.form['comment']
        toxic, severe_toxic, obsence, threat, insult, identity_hate = rate_toxic(txt_input)
        '''
        response = {}
        response['Toxic score'] = '%.4f'%toxic
        response['Severe toxic score'] = '%.4f'%severe_toxic
        response['Obsence score'] = '%.4f'%obsence
        response['Threat score'] = '%.4f'%threat
        response['Insult score'] = '%.4f'%insult
        response['Identity hate score'] = '%.4f'%identity_hate
        '''
        return render_template('home.html', Score1='%.4f'%toxic, Score2='%.4f'%severe_toxic, Score3='%.4f'%obsence, Score4='%.4f'%threat, Score5='%.4f'%insult, Score6='%.4f'%identity_hate)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port="5000")

















