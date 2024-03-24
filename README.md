# Exploratory Data Analysis: Analysis of Factors Influencing Bank Customer Churn

## ``` PROBLEM BACKGROUND ```

Anda adalah seorang data analyst di sebuah Bank Berlian. 
Departemen sales menemukan bahwa terdapat pelanggan yang tiba-tiba berhenti menggunakan layanan produk yang ada (churn 20%). 
Ini menimbulkan kekhawatiran karena dapat berdampak negatif pada pendapatan dan reputasi perusahaan.
Untuk mengatasi masalah ini, Anda diminta untuk melakukan analisis perilaku pelanggan (Churn). 
Hasil analisis Anda akan digunakan untuk menentukan rekomendasi yang tepat bagi perusahaan dalam menghadapi tantangan ini.

``` OBJEKTIF ```
1. Temukan segmentasi dengan tingkat churn paling tinggi.
2. Melihat tingkat hubungan faktor lain terhadap tingkat churn.
3. Berikan rekomendasi berdasarkan hasil analisis churn.

``` TUJUAN ```
- Tujuan bisnis untuk mengurangi churn sebesar 35%.

Dataset yang digunakan adalah "Bank Customer Churn Prediction" yang berisi:
- Skor kredit [score_credit]
- Negara [country]
- Jenis kelamin [gender]
- Usia [age]
- Masa kerja (tenure) [tenure]
- Saldo [balance]
- Jumlah produk yang dimiliki [products_number]
- Kepemilikan kartu kredit [credit_card]
- Keanggotaan aktif [active_member]
- Pendapatan perkiraan [estimated_salary]
- Status churn [churn]

Analisis dilakukan untuk memahami faktor-faktor apa yang mempengaruhi kecenderungan churn nasabah bank. Metode analisis yang digunakan termasuk uji chi-square untuk variabel kategorikal dan regresi logistik untuk variabel numerik terhadap churn. Tujuan akhirnya adalah untuk mengidentifikasi pola dan tren yang dapat membantu bank dalam memprediksi dan mencegah churn nasabah, serta meningkatkan retensi nasabah.

## ```DATA CLEANING```
1. **Handling Missing Values:** Menghilangkan atau mengisi nilai yang hilang dalam dataset untuk memastikan konsistensi dan keakuratan analisis.
2. **Handling Duplicates:** Mendeteksi dan menghapus duplikasi data untuk mencegah bias dan menjaga integritas data.
3. **Handling Outliers:** Mengatasi outlier dalam data untuk mencegah mereka memengaruhi statistik deskriptif dan model analisis.
Ini adalah langkah-langkah penting yang memastikan data yang digunakan untuk analisis selanjutnya bersih, konsisten, dan dapat diandalkan.



# **1. segmentasi dengan tingkat churn paling tinggi**
# ```EXPLONATORY VARIABEL```
## `DESKRIPSI STATISTIK`

