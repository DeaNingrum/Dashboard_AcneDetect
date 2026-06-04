## Dataset

Dataset gambar jerawat yang digunakan pada proyek ini berasal dari Kaggle:

**Acne Severity Classification Dataset**
Link: https://www.kaggle.com/datasets/lexuanhieu131297/acne-severity-classification

Dataset tidak disertakan langsung ke dalam repository GitHub karena berisi banyak file gambar dengan ukuran yang cukup besar. Jika seluruh dataset gambar dimasukkan ke repository, ukuran repository akan menjadi terlalu besar dan kurang efisien untuk proses upload, clone, maupun deployment.

Oleh karena itu, repository ini hanya menyertakan source code, notebook pengolahan data, file metadata hasil pemrosesan, serta dokumentasi proyek. Untuk menjalankan proses data preparation atau training model dari awal, pengguna dapat mengunduh dataset secara manual melalui link Kaggle di atas.

Setelah dataset diunduh, letakkan folder dataset sesuai struktur yang digunakan pada notebook, misalnya:

```text
data/
└── Acnedataset/
    ├── JPEGImages/
    ├── NNEW_trainval_0.txt
    ├── NNEW_trainval_1.txt
    ├── NNEW_trainval_2.txt
    ├── NNEW_trainval_3.txt
    ├── NNEW_test_0.txt
    ├── NNEW_test_1.txt
    ├── NNEW_test_2.txt
    └── NNEW_test_3.txt
```

Catatan: struktur folder dapat disesuaikan dengan lokasi dataset pada perangkat masing-masing. Jika path dataset berbeda, sesuaikan kembali bagian `filepath` pada notebook sebelum menjalankan proses pengolahan data.
