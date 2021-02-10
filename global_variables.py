genres = 'blues classical country disco hiphop jazz metal pop reggae rock electronic easy-listening'.split()

playlists = '''7qACZGMjyo64TdUdKAegjp 3HYK6ri0GkvRcM6GkKh0hJ
            4mijVkpSXJziPiOrK7YX4M 0ZVSWcJIf7cvycEn9HUvps 6MXkE0uYF4XwU4VTtyrpfP
            5EyFMotmvSfDAZ4hSdKrbx 3pBfUFu8MkyiCYyZe849Ks 6gS3HhOiI17QNojjPuPzqc
            0TcXdt4sbITbwCwwFbKYyd 7dowgSWOmvdpwNkGFMUs6e 6I0NsYzfoj7yHXyvkZYoRx
            47RSt6JBM6AbqBxtggW1Jk'''.split()

columns = ['genre', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
           'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

db_file = 'datasets/songs_main_genres.csv'
model_file = 'models/model.pkl'
