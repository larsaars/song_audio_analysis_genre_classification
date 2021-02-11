"""
create database by collecting song features from playlists which we know the genre of
"""

import pandas as pd
import global_variables as gv
from login_spotify import sp
import numpy as np

to_append = []

for g in zip(gv.genres, gv.playlists):

    # print progress
    print(g[0])

    # get playlist for given genre (default limit is 100)
    track_ids = sp.playlist_items(g[1],
                                  fields='items(track(id))', limit=100)
    # some lists for gathering data
    tracks_list = []

    for track in track_ids['items']:
        tracks_list.append(track['track']['id'])

    # get audio features for 50 tracks at a time (spotipy only allows 50 at once)
    # code below can be looped if more tracks need to be analyzed
    for track_idx in range(len(tracks_list)):
        try:
            bars = sp.audio_analysis(tracks_list[track_idx])['bars']
            confidences_collected = []

            # loop through all bars
            for bar in bars:
                confidences_collected.append(bar['confidence'])

            arr_len = len(confidences_collected)
            group = arr_len // 45
            leftover = arr_len - (arr_len % group)
            confidences = np.array(confidences_collected[:leftover]).reshape(-1, group).mean(axis=1)

            to_append.append([g[0]] + confidences.tolist())
        except:
            pass
# create the data frame and save
pd.DataFrame(to_append).to_csv(gv.db_file, sep=';', index=False)
