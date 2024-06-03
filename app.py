from utilities import inference, NotePredictionModel
from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer

app = Flask(__name__)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = NotePredictionModel()
state_dict = torch.load('./files/model_state_dict.pth')
model.load_state_dict(state_dict)
model.eval()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    title = request.form.get('title')
    overview = request.form.get('overview')
    genres = request.form.get('genres')

    note = inference(model=model,
                     tokenizer=tokenizer,
                     title=title,
                     overview=overview,
                     genres=genres)

    return jsonify({'Your predicted average vote': note})


if __name__ == '__main__':
    app.run(debug=True)