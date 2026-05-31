# AcneVision AI - Data Science Analytics Dashboard

AcneVision AI Dashboard adalah aplikasi analitik berbasis web yang dirancang untuk memvisualisasikan performa model, kualitas dataset, analisis dampak ekonomi, serta evaluasi keadilan (*algorithmic fairness*) dari sistem klasifikasi tingkat keparahan jerawat.

Aplikasi ini dibangun menggunakan Streamlit dengan antarmuka kustom yang mendukung pergantian tema dinamis (*Light/Dark mode*) serta fitur penyaringan data (*slicing parameter*) secara real-time.

---

## Fitur Utama Dashboard

Dashboard ini dibagi menjadi 4 modul analisis utama yang dapat diakses melalui menu navigasi pada sidebar:

### 1. Business Impact & Overview
* **Metrik Eksekutif:** Menampilkan total sampel hasil filter, akurasi dasar model (94.2%), rata-rata skor keyakinan (*confidence score*), dan kelas keparahan yang dominan.
* **Distribusi Interaktif:** Grafik *box plot* rentang keyakinan model berdasarkan tingkat keparahan jerawat dan *pie chart* proporsi kelas data.
* **Dampak Ekonomi:** Proyeksi penghematan biaya konsumen akibat pencegahan salah diagnosis serta indikator kelayakan integrasi produk ke pasar (*Production-ready*).

### 2. Dataset Quality & EDA
* **Validasi Pembagian Data:** Visualisasi rasio pembagian data *Train/Validation* dan *Test set*.
* **Kontrol Kualitas Citra:** Scatter plot untuk menganalisis karakteristik teknis citra berdasarkan nilai kecerahan (*brightness*) dan indeks keburaman (*blur score*).
* **Matriks Korelasi:** Heatmap interaktif untuk menguji multikolinearitas antar metrik kualitas citra guna memastikan model tidak bias terhadap pencahayaan gambar.

### 3. Model Evaluation & Diagnostics
* **Metrik Performa Klasifikasi:** Menampilkan pencapaian skor *Precision* (93.1%), *Recall* (92.6%), *F1-Score* (92.8%), dan *AUC-ROC* (0.965).
* **Confusion Matrix:** Diagnostik visual mendalam untuk memetakan prediksi model terhadap label aktual data uji.
* **Analisis Defisit Prediksi:** Diagram lingkaran (*pie chart*) pemecahan galat (*error breakdown*) antara *False Positive*, *False Negative*, dan data pencilan (*outlier*).
* **Kurva Lanjutan:** Grafik riwayat *Loss* selama pelatihan (Training vs Validation) dan kurva karakteristik operasi penerima (ROC) per kelas severitas.

### 4. Fairness & Recommendations
* **Pemetaan Bahan Aktif:** Analisis distribusi bahan aktif kosmetik (seperti *Salicylic Acid*, *Niacinamide*, *Retinol*) yang direkomendasikan sistem berdasarkan kondisi kulit pengguna.
* **Uji Keadilan Algoritma:** Pengujian kestabilan akurasi model AI lintas demografi warna kulit (berdasarkan Skala Fitzpatrick) untuk memastikan sistem bebas dari bias genetik.

---

## Spesifikasi Teknologi (Tech Stack)

* **Framework Antarmuka:** Streamlit
* **Manipulasi Data:** Pandas, NumPy
* **Visualisasi Interaktif:** Plotly Express, Plotly Graph Objects
* **Grafik Statistik:** Seaborn, Matplotlib
* **Integrasi Tipografi:** Plus Jakarta Sans (via Google Fonts API)

---

## Struktur Direktori Proyek

Pastikan direktori Anda terstruktur seperti di bawah ini agar aplikasi dapat membaca berkas data dengan benar:

```text
📂 nama-repositori-anda/
│
├── 📄 app.py                         # Berkas kode utama Streamlit
├── 📄 requirements.txt               # Daftar dependensi pustaka Python
├── 📄 README.md                      # Dokumentasi proyek
│
└── 📂 data/                          # Folder penyimpanan data (Rekomendasi)
    ├── 📊 metadata_acne_final.csv    # Dataset karakteristik citra & model jerawat
    └── 📊 dataset_skincare_ready.csv  # Dataset produk & rekomendasi bahan aktif