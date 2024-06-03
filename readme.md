# Movie vote predictor

This project aims to build a model which predicts the average vote a movie would have, given its title, a short overview, and its genres. 

It was trained on data exctracted from the following dataset: https://huggingface.co/datasets/AiresPucrs/tmdb-5000-movies, which contains information of 5000 movies extracted from the [TMDB](https://www.themoviedb.org/) website. The preprocessing part can be found in the 'preprocess_df.py' file. The preprocessed dataframe can be found in 'files', under the name 'preprocessed_df.csv'.

The model uses the [BERT](https://huggingface.co/google-bert/bert-base-uncased) language model, with LoRA adapters, for processing the title and overview, on top of which was added a neural network head to process the genres, which were encoded as a vector containing 1 if the movie had the associated genre, and 0 if not.  The target is an average vote, ranging from 0 to 10. The training can be found in the 'train.py' file, which makes use of several functions and classes contained in the 'utilities.py' file. 

Finally, I created a Flask application, which enables to test the model. It uses an HTML template which can be found in 'templates/index.html'. The application loads the model via a state dictionnary that I saved when training was finished, which can be found in 'files/model_state_dict.pth'. 

### If you wish to train this model locally

The model was trained on my computer, with an apple sillicon gpu. You can change the device in the 'train.py' file simply by replacing the line:

```python
device = torch.device('mps')
```
with your device.

### Requirements

Requirements for running this project can be found in the 'requirements.txt' file.


