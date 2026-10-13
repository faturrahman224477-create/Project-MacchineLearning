import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pypdf import PdfReader

# Konfigurasi Halaman Web
st.set_page_config(page_title="AI Resume Matcher & Career Lab", page_icon="🤖", layout="wide")

st.title("🤖 AI Resume & Portfolio Matching Engine HRD FATUR")
st.write("Lab interaktif berbasis Machine Learning untuk evaluasi kesesuaian CV dengan standar industri secara instan.")

st.markdown("---")

# Data Template Deskripsi Pekerjaan Berdasarkan Kategori
job_templates = {
    "Data Scientist": "Mencari data scientist yang menguasai Python, Machine Learning, SQL, Pandas, Scikit-Learn, dan visualisasi data.",
    "Software Engineer": "Mencari software engineer berpengalaman yang menguasai Git, REST API, Database SQL/NoSQL, JavaScript, Python, dan arsitektur sistem.",
    "Digital Marketer": "Mencari digital marketer yang handal dalam SEO, SEM, Google Ads, Meta Ads, analisis trafik, content strategy, dan copywriter.",
    "UI/UX Designer": "Mencari UI/UX designer yang menguasai Figma, wireframing, user research, prototyping, dan desain interaksi."
}

# Sidebar untuk Pengaturan & Kategori
st.sidebar.header("⚙️ Pengaturan Evaluasi")
mode_input = st.sidebar.radio("Pilih Sumber Lowongan Kerja:", ["Pilih Kategori Standar", "Tulis Manual / Custom"])

if mode_input == "Pilih Kategori Standar":
    selected_role = st.sidebar.selectbox("Pilih Kategori Pekerjaan:", list(job_templates.keys()))
    default_jd = job_templates[selected_role]
else:
    default_jd = "Tulis atau tempel deskripsi pekerjaan di sini..."

# Layout Utama (2 Kolom: Kiri untuk Lowongan, Kanan untuk Upload CV)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Deskripsi Pekerjaan (Job Description)")
    job_description = st.text_area("Detail lowongan kerja:", value=default_jd, height=250)

with col2:
    st.subheader("📄 Unggah CV (Format PDF)")
    uploaded_file = st.file_uploader("Pilih file CV PDF kamu...", type=["pdf"])
    
    candidate_cv = ""
    if uploaded_file is not None:
        try:
            # Membaca teks dari file PDF yang diupload
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    candidate_cv += text + "\n"
            st.success(f"✅ Berhasil membaca CV: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Gagal membaca file PDF: {e}")
    
    # Opsi cadangan jika ingin ketik/tempel teks manual
    with st.expander("Atau tempel teks CV secara manual (Opsional)"):
        manual_cv = st.text_area("Teks CV manual:", value="Saya adalah seorang pemrogram yang terbiasa menggunakan Python dan Pandas.")
        if manual_cv and not candidate_cv:
            candidate_cv = manual_cv

st.markdown("---")

# Tombol Eksekusi Analisis
if st.button("🚀 Analisis Kecocokan CV Sekarang", type="primary", use_container_width=True):
    if job_description.strip() == "" or candidate_cv.strip() == "":
        st.warning("⚠️ Mohon pastikan Deskripsi Pekerjaan sudah terisi dan file CV PDF sudah diunggah!")
    else:
        # Proses TF-IDF & Cosine Similarity
        documents = [job_description, candidate_cv]
        vectorizer = TfidfVectorizer(stop_words='english') # Mengabaikan kata sambung umum
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        score_percentage = similarity_score[0][0] * 100

        # Tampilkan Hasil dalam Layout Kolom
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown("### 📊 Skor Kecocokan")
            st.metric(label="Match Score", value=f"{score_percentage:.2f}%")
            st.progress(int(min(score_percentage, 100)))

        with res_col2:
            # Analisis Keyword Gap
            feature_names = vectorizer.get_feature_names_out()
            jd_vector = tfidf_matrix[0].toarray()[0]
            cv_vector = tfidf_matrix[1].toarray()[0]

            keyword_df = pd.DataFrame({'kata': feature_names, 'bobot_jd': jd_vector, 'bobot_cv': cv_vector})
            missing_keywords = keyword_df[(keyword_df['bobot_jd'] > 0) & (keyword_df['bobot_cv'] == 0)]

            st.markdown("### 🔍 Kata Kunci yang Perlu Ditambahkan ke CV:")
            if not missing_keywords.empty:
                st.error("Sistem mendeteksi kata kunci penting dari lowongan berikut belum ditemukan di CV kamu:")
                # Tampilkan sebagai badge/tags rapi
                missing_words_list = [f"`{word}`" for word in missing_keywords['kata'].values]
                st.markdown(" ".join(missing_words_list))
            else:
                st.success("🎉 Sempurna! Seluruh kata kunci utama dari lowongan sudah tercakup di dalam CV kamu.")
