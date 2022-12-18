from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import csv

"""
pip install spotipy
export SPOTIPY_CLIENT_ID=4355e1e5d95c40099d5759362c8db140
export SPOTIPY_CLIENT_SECRET=5788e8b63c8f4f1a83a4b8cb05381229
export SPOTIPY_REDIRECT_URI=https://localhost:8888/callback
"""

client_credentials_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

BLUES_ID = '37i9dQZF1DXd9rSDyQguIk'
CLASSICAL_ID = '1h0CEZCm6IbFTbxThn6Xcs'
JAZZ_ID = '37i9dQZF1DXbITWG1ZJKYt'
METAL_ID = '1GXRoQWlxTNQiMNkOe7RqA'
POP_ID = '6mtYuOxzl58vSGnEDtZ9uB'
DISCO_ID = '2iUm4Ez2UGUpdN4KuBtAu0'
ROCK_ID = '37i9dQZF1DWXRqgorJj26U'
COUNTRY_ID = '37i9dQZF1DWZBCPUIUs2iR'
HIPHOP_ID = '3RcRK9HGTAm9eLW1LepWKZ'
REGGAE_ID = '4ONdTgODsdMvCrJ9ANld3Y'

LABEL = ["blues", "classical", "rock", "metal", "pop", "disco", "jazz", "country", "hiphop", "reggae"]
ALL_ID = [BLUES_ID, CLASSICAL_ID, ROCK_ID, METAL_ID, POP_ID, DISCO_ID, JAZZ_ID, COUNTRY_ID, HIPHOP_ID, REGGAE_ID]
INTERESSANT_KEYS = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

with open('data/data_spotify.csv', 'w', newline='') as csvfile:
    fieldnames = INTERESSANT_KEYS.copy()
    fieldnames.append("genre")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for playlist_id, label in zip(ALL_ID, LABEL):
        results = sp.playlist(playlist_id)
        uris = [x["track"]["uri"] for x in results["tracks"]["items"][:100]]
        features = sp.audio_features(uris)
        for feature in features:
            dico = {x : feature[x] for x in INTERESSANT_KEYS}
            dico["genre"] = label
            writer.writerow(dico)
