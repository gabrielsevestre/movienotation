from datasets import load_dataset
import pandas as pd
import json
import re


list_of_genres = ['Crime', 'Comedy', 'Adventure', 'Action', 'Science Fiction',
                  'Animation', 'Family', 'Drama', 'Romance', 'Music', 'Fantasy',
                  'Thriller', 'War', 'Western', 'Mystery', 'History', 'Horror',
                  'Documentary', 'Foreign', 'TV Movie']

dataset = load_dataset("AiresPucrs/tmdb-5000-movies")
df = dataset['train'].to_pandas()

df = df.drop([4590, 4770, 4802])  # no overview

df = df.drop(columns=['id', 'homepage', 'cast', 'crew', 'budget', 'production_companies',
                      'keywords', 'release_date', 'revenue', 'runtime', 'popularity',
                      'spoken_languages', 'original_language', 'original_title', 'tagline',
                      'texts', 'original_title', 'tagline'])
df = df.drop(df[df['status'] != 'Released'].index)
df = df.drop(columns='status')
df = df.drop(df[df['vote_count'] < 10].index)
df = df.drop(columns='vote_count')


def process_genres_and_countries(row):
    g = json.loads(row['genres'])
    c = json.loads(row['production_countries'])
    genres = [gg['name'] for gg in g]
    countries = [cc['name'] for cc in c]
    return genres, countries


gc = df.apply(process_genres_and_countries, axis=1, result_type='expand')
df[['genre', 'countries']] = gc  # The 'genre' column was used to get the 'list_of_genres' at the beginning
df = df.drop(columns=['genres', 'production_countries'])
df = df.drop(columns='countries')  # too much US, won't be coherent

df = df.fillna(value='')


def process_text_and_len(row):
    title = row['title'].lower()
    overview = row['overview'].lower()
    title = re.sub(r'\s+', ' ', title)
    title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
    overview = re.sub(r'\s+', ' ', overview)
    overview = re.sub(r'[^a-zA-Z0-9\s]', '', overview)
    return title, overview


newcol = df.apply(process_text_and_len, axis=1, result_type='expand')
df[['title', 'overview']] = newcol

exploded = df.explode('genre')
dummies = pd.get_dummies(exploded['genre'])
combined = pd.concat([exploded, dummies], axis=1)
result = combined.groupby(exploded.index).sum()
df = df.join(result[list_of_genres])
df = df.drop(columns='genre')

# df.to_csv('preprocessed_df.csv')
