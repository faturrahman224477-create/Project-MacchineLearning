import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Konfigurasi Halaman Web
st.set_page_config(page_title="ATS Resume Matcher Engine", page_icon="🤖", layout="centered")

st.title("🤖 Automated Resume & Portfolio Matching Engine")
st.write("Lab interaktif berbasis Machine Learning untuk mengukur kecocokan CV dengan Lowongan Kerja sekaligus mendeteksi kata kunci yang kurang.")

st.markdown("---")

# Layout Input Teks
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Deskripsi Pekerjaan (Job Desc)")
    job_description = st.text_area("Tempel teks lowongan kerja di sini...", height=200, 
                                   value="Mencari data scientist yang menguasai Python, Machine Learning, SQL, dan Pandas.")

with col2:
    st.subheader("📄 Teks CV / Portofolio")
    candidate_cv = st.text_area("Tempel isi CV kamu di sini...", height=200,
                                value="Saya adalah seorang pemrogram yang terbiasa menggunakan Python dan Pandas untuk analisis data serta membuat model machine learning.")

# Tombol Eksekusi Analisis
if st.button("🚀 Analisis Kecocokan CV", type="primary"):
    if job_description.strip() == "" or candidate_cv.strip() == "":
        st.warning("⚠️ Mohon isi kedua kolom teks terlebih dahulu!")
    else:
        # Proses TF-IDF & Cosine Similarity
        documents = [job_description, candidate_cv]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        score_percentage = similarity_score[0][0] * 100

        # Tampilkan Hasil Skor
        st.markdown("### Hasil Evaluasi")
        st.metric(label="Skor Kecocokan (Match Score)", value=f"{score_percentage:.2f}%")
        
        # Progress bar visual
        st.progress(int(score_percentage))

        # Analisis Keyword Gap
        feature_names = vectorizer.get_feature_names_out()
        jd_vector = tfidf_matrix[0].toarray()[0]
        cv_vector = tfidf_matrix[1].toarray()[0]

        keyword_df = pd.DataFrame({'kata': feature_names, 'bobot_jd': jd_vector, 'bobot_cv': cv_vector})
        missing_keywords = keyword_df[(keyword_df['bobot_jd'] > 0) & (keyword_df['bobot_cv'] == 0)]

        st.markdown("### 🔍 Kata Kunci Penting yang Belum Ada di CV:")
        if not missing_keywords.empty:
            st.error("Sistem mendeteksi beberapa kata kunci dari lowongan yang belum tercantum di CV kamu:")
            for word in missing_keywords['kata'].values:
                st.write(f"- `{word}`")
        else:
            st.success("🎉 Luar biasa! Semua kata kunci penting dari lowongan sudah ada di dalam CV kamu.")
