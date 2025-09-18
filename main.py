from transformers import pipeline
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

def get_label(result):
    return  max(result, key=lambda x: x["score"])["label"]

def get_spectrogramm(audio_path):
    y, sr = librosa.load(audio_path)
    print("Creating mel-spectogramm")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")

    x_mel = librosa.feature.melspectrogram(y=y, sr=sr)
    x_mel = librosa.power_to_db(x_mel, ref=np.max)

    librosa.display.specshow(x_mel, sr=sr, x_axis="time", y_axis="mel")
    plt.savefig('spectogramm.png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

def audio_classifire(pipe):
    result = pipe("spectogramm.png")
    return get_label(result)

def get_name(path):
    return path.split("/")[-1].split(".")[0]

def recommend_tracks_from_parquet(model, user_genre, user_artist, emb_matrix, top_k=5):
    query_text = f"genre: {user_genre}, artist: {user_artist}"
    query_emb = model.encode(query_text, normalize_embeddings=True)
    scores = np.dot(emb_matrix, query_emb)
    top_indices = np.argsort(-scores)[:top_k]
    return df.iloc[top_indices][["track_name", "artist_name", "genre", "popularity"]]

def print_recomendation(recomendation):
    print("Попробуй послушать еще такое")
    for i, row in recomendation.iterrows():
        name = row["track_name"]
        artist = row["artist_name"]
        genre = row["genre"]
        popularity = row["popularity"]

        print(f"\"{name}\" — {artist}")
        print(f"   Жанр: {genre} | Популярность: {popularity}")


print("Preparing audio classifire")
image_pipe = pipeline("image-classification", "sound_classification_model")

print("Preparing embedding model")
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

print("Preparing embedding dataset")
df = pd.read_parquet("spotify_tracks_embeddings.parquet")
emb_matrix = np.array(df["embeddings"].tolist(), dtype=np.float32)
emb_matrix /= np.linalg.norm(emb_matrix, axis=1, keepdims=True)

print("Ready for work")


while True:
    path = input("Enter audio path: ")
    get_spectrogramm(path)
    genre = audio_classifire(image_pipe)
    name = get_name(path)
    print(f"\nPredictet label: {genre}\n")
    recommendation = recommend_tracks_from_parquet(embedding_model, genre, name, emb_matrix)
    print_recomendation(recommendation)
