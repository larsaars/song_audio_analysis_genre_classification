# load credentials
import spotipy

with open('spotify_secrets.txt', 'r') as file:
    content = file.read().splitlines()
    client_id = content[0]
    client_secret = content[1]

# login
sp = spotipy.Spotify(auth_manager=spotipy.SpotifyClientCredentials(client_id=client_id,
                                                                   client_secret=client_secret))