import pandas as pd
import random

# ================= LOAD DATA =================
df = pd.read_csv("data/user_music_rating_dataset.csv")

# Jumlah user simulasi
NUM_USERS = 50
RATINGS_PER_USER = 10  # tiap user rating 10 lagu

users = [f"U{i:03d}" for i in range(1, NUM_USERS + 1)]
rows = []

for user in users:
    sampled_songs = df.sample(
        n=RATINGS_PER_USER,
        random_state=random.randint(1, 9999)
    )

    for _, row in sampled_songs.iterrows():
        rows.append({
            "user_id": user,
            "track_name": row["track_name"],
            "rating": row["rating"]
        })

dense_df = pd.DataFrame(rows)

# ================= SAVE =================
dense_df.to_csv("data/user_music_rating_dense.csv", index=False)

print("✅ Dataset padat berhasil dibuat")
print("Ukuran dataset:", dense_df.shape)
