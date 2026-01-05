import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="MusicMate 🎧",
    layout="wide"
)

# ================= STYLE (OCEAN BLUE) =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #e6f4ff, #f8fbff);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a3d62, #1e6fa3);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
input, textarea, select {
    border-radius: 12px !important;
}
button {
    background: linear-gradient(135deg,#1e90ff,#0a3d62)!important;
    color:white!important;
    border-radius:14px!important;
    font-weight:bold!important;
}
.ocean-card {
    padding:16px;
    margin-bottom:14px;
    border-radius:18px;
    background:linear-gradient(135deg,#1e90ff,#48dbfb);
    color:white;
    box-shadow:0 6px 18px rgba(0,0,0,.15);
}
</style>
""", unsafe_allow_html=True)

# ================= DATA =================
DATA_PATH = "data/user_music_rating_dataset.csv"
df = pd.read_csv(DATA_PATH)

df["user_id"] = df["user_id"].astype(str)
df["track_name"] = df["track_name"].astype(str)

if "timestamp" not in df.columns:
    df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(DATA_PATH, index=False)

# ================= SESSION =================
st.session_state.setdefault("login", False)
st.session_state.setdefault("user_id", None)

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## 🎧 MusicMate")
    st.caption("Rekomendasi Musik Personal")

    mode = st.radio("Mode", ["Login", "Register"])

    if mode == "Login":
        uid = st.text_input("User ID", placeholder="contoh: U001")
        if st.button("Masuk"):
            if uid in df["user_id"].values:
                st.session_state.login = True
                st.session_state.user_id = uid
                st.success("Login berhasil")
            else:
                st.error("User tidak ditemukan")

    else:
        if st.button("Daftar User Baru"):
            df["num"] = df["user_id"].str.extract(r'(\d+)').astype(int)
            new_id = f"U{df['num'].max()+1:03d}"
            st.session_state.login = True
            st.session_state.user_id = new_id
            st.success(f"User dibuat: {new_id}")

    if st.session_state.login:
        st.markdown("---")
        st.success(f"Login: {st.session_state.user_id}")
        if st.button("Logout"):
            st.session_state.login = False
            st.session_state.user_id = None
            st.rerun()

# ================= MAIN =================
st.title("🎵 MusicMate – Sistem Rekomendasi Musik")

if not st.session_state.login:
    st.info("Silakan login atau register")
    st.stop()

user_df = df[df["user_id"] == st.session_state.user_id]

# ================= INPUT RATING =================
st.subheader("⭐ Tambah Rating Lagu")

with st.form("rating_form", clear_on_submit=True):
    search = st.text_input("🔍 Cari Lagu")

    song_list = sorted(df["track_name"].unique())
    filtered = [s for s in song_list if search.lower() in s.lower()] if search else song_list

    song = st.selectbox("🎵 Pilih Lagu", filtered)
    rating = st.slider("⭐ Rating", 1, 10, 5)

    submit = st.form_submit_button("💾 Simpan Rating")

if submit:
    new_row = pd.DataFrame([{
        "user_id": st.session_state.user_id,
        "track_name": song,
        "rating": rating,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

    st.success("✅ Rating berhasil disimpan")
    st.rerun()

# ================= EDIT & DELETE =================
st.subheader("✏️ Edit / 🗑️ Hapus Rating")

user_df = df[df["user_id"] == st.session_state.user_id]

if not user_df.empty:
    idx = st.selectbox(
        "Pilih Lagu",
        user_df.index,
        format_func=lambda i: f"{user_df.loc[i,'track_name']} (⭐ {user_df.loc[i,'rating']})"
    )

    new_rating = st.slider(
        "Ubah Rating",
        1, 10,
        int(user_df.loc[idx, "rating"])
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✏️ Update"):
            df.loc[idx, "rating"] = new_rating
            df.to_csv(DATA_PATH, index=False)
            st.success("Rating diperbarui")
            st.rerun()

    with col2:
        if st.button("🗑️ Hapus"):
            df = df.drop(idx)
            df.to_csv(DATA_PATH, index=False)
            st.warning("Rating dihapus")
            st.rerun()
else:
    st.info("Belum ada rating")

# ================= GRAFIK BAR =================
st.subheader("📊 Distribusi Rating Kamu")

if not user_df.empty:
    rating_count = user_df["rating"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    rating_count.plot(kind="bar", ax=ax)

    ax.set_xlabel("Rating")
    ax.set_ylabel("Jumlah Lagu")
    ax.set_title("Distribusi Rating User")
    ax.tick_params(axis="x", rotation=0)

    st.pyplot(fig)
else:
    st.info("Belum ada data rating")

# ================= REKOMENDASI KNN (FIX AMAN) =================
st.subheader("🧠 Rekomendasi Untuk Kamu")

user_item = df.pivot_table(
    index="user_id",
    columns="track_name",
    values="rating",
    fill_value=0
)

if st.session_state.user_id not in user_item.index:
    st.info("Beri minimal 1 rating dulu untuk mendapatkan rekomendasi 🎵")
    st.stop()

if len(user_item) < 2:
    st.info("Belum cukup data user untuk rekomendasi")
    st.stop()

knn = NearestNeighbors(metric="cosine", algorithm="brute")
knn.fit(user_item)

n_neighbors = min(4, len(user_item))

dist, idx = knn.kneighbors(
    user_item.loc[[st.session_state.user_id]],
    n_neighbors=n_neighbors
)

similar_users = user_item.index[idx.flatten()[1:]]

scores = {}
for u in similar_users:
    for song, r in user_item.loc[u].items():
        if user_item.loc[st.session_state.user_id][song] == 0 and r > 0:
            scores[song] = scores.get(song, 0) + r

if not scores:
    st.info("Belum ada rekomendasi")
else:
    for song, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]:
        st.markdown(f"""
        <div class="ocean-card">
        🎵 <b>{song}</b><br>
        🔮 Skor rekomendasi: {score}
        </div>
        """, unsafe_allow_html=True)
