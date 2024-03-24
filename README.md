# bank_churn

## ``` PROBLEM BACKGROUND ```

Anda adalah seorang data analyst di sebuah Bank Berlian. 
Departemen sales menemukan bahwa terdapat pelanggan yang tiba-tiba berhenti menggunakan layanan produk yang ada (churn 20%). 
Ini menimbulkan kekhawatiran karena dapat berdampak negatif pada pendapatan dan reputasi perusahaan.
Untuk mengatasi masalah ini, Anda diminta untuk melakukan analisis perilaku pelanggan (Churn). 
Hasil analisis Anda akan digunakan untuk menentukan rekomendasi yang tepat bagi perusahaan dalam menghadapi tantangan ini.

``` OBJEKTIF ```
- Temukan segmentasi dengan tingkat churn paling tinggi.
- Melihat tingkat hubungan faktor lain terhadap tingkat churn.
- Berikan rekomendasi berdasarkan hasil analisis churn.

``` TUJUAN ```
- Tujuan bisnis untuk mengurangi churn sebesar 35%.

Dataset yang digunakan adalah "Bank Customer Churn Prediction" yang berisi:
1. Skor kredit [score_credit]
2. Negara [country]
3. Jenis kelamin [gender]
4. Usia [age]
5. Masa kerja (tenure) [tenure]
6. Saldo [balance]
7. Jumlah produk yang dimiliki [products_number]
8. Kepemilikan kartu kredit [credit_card]
9. Keanggotaan aktif [active_member]
10. Pendapatan perkiraan [estimated_salary]
11. Status churn [churn]

Analisis dilakukan untuk memahami faktor-faktor apa yang mempengaruhi kecenderungan churn nasabah bank. Metode analisis yang digunakan termasuk uji chi-square untuk variabel kategorikal dan regresi logistik untuk variabel numerik terhadap churn. Tujuan akhirnya adalah untuk mengidentifikasi pola dan tren yang dapat membantu bank dalam memprediksi dan mencegah churn nasabah, serta meningkatkan retensi nasabah.

## ```Data Cleaning```
1. **Handling Missing Values:** Menghilangkan atau mengisi nilai yang hilang dalam dataset untuk memastikan konsistensi dan keakuratan analisis.
2. **Handling Duplicates:** Mendeteksi dan menghapus duplikasi data untuk mencegah bias dan menjaga integritas data.
3. **Handling Outliers:** Mengatasi outlier dalam data untuk mencegah mereka memengaruhi statistik deskriptif dan model analisis.
Ini adalah langkah-langkah penting yang memastikan data yang digunakan untuk analisis selanjutnya bersih, konsisten, dan dapat diandalkan.
