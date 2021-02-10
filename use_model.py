"""
use the model, search for songs
"""

import spotipy
from sklearn.ensemble import RandomForestClassifier
from login_spotify import sp
import pickle as pkl
import numpy as np
import global_variables as gv

# load model
print('<< loading model...')
model: RandomForestClassifier
with open(gv.model_file, 'rb') as file:
    model = pkl.load(file)

# print when done loading
print('<< Done loading! Enter a song name or \'q\' to quit:')

# loop prediction for input
while True:
    inp = input('>> ')

    # break if input is 'q'
    if inp == 'q':
        break

    # get first result
    song_results = sp.search(q=inp, limit=1)

    # call in loop for auto skipping if no results
    for idx, track in enumerate(song_results['tracks']['items']):
        # print the name and artist of track
        artists = []
        for idx2, artist in enumerate(track['artists']):
            artists.append(artist['name'])

        print('<< %s - %s' % (track['name'], artists[0] if len(artists) == 1 else str(artists)))
        # else predict from features
        # get the features
        ft = sp.audio_features(track['id'])[0]
        # extract to list to be predictable
        ft_list = [ft['danceability'], ft['energy'], ft['key'],
                   ft['loudness'], ft['mode'],
                   ft['speechiness'], ft['acousticness'],
                   ft['instrumentalness'], ft['liveness'],
                   ft['valence'], ft['tempo']]

        # predict probabilities
        probs = model.predict_proba([ft_list])[0]

        # return probs
        max_idx = np.argsort(-probs)[:3]
        # print every idx
        for idx3 in range(len(max_idx)):
            print('<< %i: %s [%f]' % (idx3 + 1, gv.genres[max_idx[idx3]], probs[max_idx[idx3]]))