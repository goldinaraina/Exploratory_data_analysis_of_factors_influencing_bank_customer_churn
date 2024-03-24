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
Analisis ini memisahkan dataset utama `df` menjadi subset yang disebut `data_churn` untuk secara khusus memfokuskan pada data churn untuk melakukan analisis lebih rinci terhadap faktor-faktor yang berkontribusi terhadap churn nasabah dalam sektor perbankan.

## ```PRE PROCESSING```
- Membuat rentang (bining) untuk mengelompokkan pelanggan ke dalam kelompok umur tertentu agar analisis lebih mudah dilakukan. Rentang pada kolom: 
- Skor kredit [score_credit]
- Usia [age]
- Saldo [balance]

## ``UNVARITE ANALISYST``
Analisis univariat merupakan pendekatan statistik yang fokus pada satu variabel tunggal pada suatu waktu. Ini membantu dalam memahami distribusi dan sifat-sifat dasar dari variabel untuk mengeksplorasi pola-pola dan karakteristik-karakteristik penting.

## `Visualisasi Data`
visualisasi data membantu dapat melihat kompleksitas data, distribusi, pola dan tren data yang ada menggunakan histogram dan barplot.

# **2. Melihat tingkat hubungan faktor lain terhadap tingkat churn**
Analisis ini menunjukkan adanya pola yang menarik terkait faktor-faktor yang memengaruhi tingkat churn pada pelanggan bank. Rentang pendapatan dan usia memainkan peran penting dalam menentukan kecenderungan churn, dengan rentang pendapatan dan usia tertentu cenderung memiliki tingkat churn yang lebih tinggi. Selain itu, penggunaan kartu kredit juga memiliki pengaruh yang signifikan, di mana pelanggan yang menggunakan kartu kredit cenderung memiliki tingkat churn yang lebih tinggi. Analisis lebih lanjut dapat dilakukan untuk memahami faktor-faktor lain yang mungkin berkontribusi terhadap tingkat churn, sehingga bank dapat mengambil langkah-langkah yang lebih tepat dalam mempertahankan pelanggan.

# ```ANALYSIS DATA```
- Odd digunakan untuk menggambarkan nilai yang berbeda atau tidak biasa dalam data.
- Korelasi untuk melihat hubungan antara dua variabel. 
- Hipotesis diajukan untuk diuji kebenarannya dalam analisis statistik. 
- Ordinary Least Squares (OLS) adalah metode yang digunakan dalam analisis regresi untuk mengestimasi parameter dalam model regresi.
- Signifikansi, khususnya dalam konteks statistik, mengindikasikan seberapa kuat bukti yang dimiliki oleh data terhadap suatu hipotesis. 
- R-Square untuk mengukur secara statistik dalam analisis regresi untuk mengevaluasi seberapa baik model regresi sesuai dengan data yang diamati.

# **3. Rekomendasi berdasarkan hasil analisis churn**
**Data Visualisasi:**
- Terdapat pola bahwa semakin tinggi rentang umur dan rentang pendapatan maka semakin tinggi pula jumlah churn yang terjadi.  Kemudian,  Jumlah churn cenderung meningkat seiring dengan peningkatan rentang skor kredit terutama pada pemilik kartu kredit sedangkan rentang skor kredit yang lebih tinggi cenderung memiliki jumlah churn yang lebih rendah. Berdasarkan visualisasi  konsumen yang memiliki kartu kredit cenderung memiliki jumlah churn yang lebih tinggi daripada yang tidak memiliki kartu kredit. Pada setiap negara, gender perempuan cenderung melakukan churn lebih tinggi dibandingkan gender laki-laki.

**Korelasi:**
- Age: 0.353756 (positif)
- Balance: 0.115052 (positif)
- Active Member: -0.145099 (negatif)
- Products Number: -0.109205 (negatif)
- Credit Card: -0.007750 (negatif)
- Tenure: -0.014092 (negatif)
- Korelasi tertinggi adalah antara usia (age) dan churn, menunjukkan hubungan positif yang cukup kuat.

**Uji Hipotesis:**

- Hipotesis 1 (H1): Terbukti bahwa usia pelanggan berpengaruh signifikan terhadap tingkat churn. Semakin tua usia pelanggan, semakin tinggi kemungkinan mereka untuk churn.
- Hipotesis 2 (H2): Terbukti bahwa saldo akun pelanggan berpengaruh signifikan terhadap tingkat churn. Semakin tinggi saldo akun pelanggan, semakin rendah kemungkinan mereka untuk churn.
- Hipotesis 3 (H3): Terbukti bahwa keterlibatan aktif pelanggan berpengaruh signifikan terhadap tingkat churn. Pelanggan yang aktif memiliki tingkat churn yang lebih rendah daripada yang tidak aktif.


berdasarkan data visualisasi, korelasi, dan uji hipotesis, kita dapat menyimpulkan bahwa faktor-faktor seperti usia, saldo akun, dan keterlibatan aktif pelanggan memiliki pengaruh yang signifikan terhadap tingkat churn dalam dataset tersebut.

## ```IN CASE TWO WEEKS```
**Relevansi:**
Analisis churn memberikan wawasan tentang perilaku pelanggan dan faktor-faktor yang memengaruhi keputusan mereka untuk tetap menggunakan layanan atau beralih ke pesaing. Dengan pemahaman ini, perusahaan dapat mengambil langkah-langkah strategis untuk meningkatkan retensi pelanggan dan mengurangi churn, yang pada gilirannya dapat meningkatkan pendapatan dan keuntungan perusahaan.

**Rekomendasi:**
1. **Segmentasi Pelanggan:** Identifikasi kelompok pelanggan berdasarkan karakteristik seperti usia, saldo akun, dan keterlibatan aktif. Ini dapat membantu dalam menyesuaikan strategi retensi yang sesuai dengan kebutuhan dan preferensi masing-masing segmen.
2. **Program Loyalty:** Sertakan program loyalitas yang menarik untuk mendorong retensi pelanggan. Program seperti penghargaan, diskon, atau penawaran khusus dapat membantu meningkatkan keterlibatan pelanggan dan mengurangi kecenderungan mereka untuk beralih.
3. **Analisis Penggunaan Produk:** Tinjau penggunaan produk oleh pelanggan dan identifikasi pola penggunaan yang menunjukkan kecenderungan churn. Berikan dukungan atau layanan tambahan kepada pelanggan yang mungkin menghadapi masalah atau kebutuhan tambahan.
4. **Komunikasi Proaktif:** Komunikasikan secara proaktif dengan pelanggan yang menunjukkan tanda-tanda potensial untuk melakukan churn. Tawarkan solusi atau bantuan yang sesuai dengan kebutuhan mereka untuk mencegah churn.
5. **Optimalkan Layanan Pelanggan:** Pastikan layanan pelanggan yang responsif dan efektif. Tanggapi pertanyaan, keluhan, atau masalah pelanggan dengan cepat dan secara memuaskan untuk meningkatkan kepuasan pelanggan.

**Peningkatan Dalam Dua Minggu:**
Berdasarkan jumlah pelanggan yang harus dipertahankan dan waktu yang tersedia (dua minggu), perlu dilakukan peningkatan sebesar [jumlah peningkatan per minggu] setiap minggu. Simulasikan langkah-langkah tertentu yang diambil untuk meningkatkan retensi pelanggan dalam dua minggu, seperti:

1. **Kampanye Retensi:** Jalankan kampanye retensi yang ditargetkan kepada pelanggan yang memiliki risiko churn tinggi.
2. **Promosi Khusus:** Tawarkan promosi khusus atau diskon kepada pelanggan yang berpotensi untuk churn untuk mendorong mereka tetap menggunakan layanan.
3. **Kegiatan Cross-Selling:** Gunakan kesempatan untuk mengenalkan produk atau layanan baru kepada pelanggan yang sudah ada untuk meningkatkan keterlibatan mereka.
4. **Peningkatan Kualitas Layanan:** Tingkatkan kualitas layanan pelanggan dengan memberikan pelatihan tambahan kepada staf dan meningkatkan responsivitas terhadap masalah pelanggan.
5. **Feedback dan Evaluasi:** Lakukan evaluasi mingguan terhadap langkah-langkah yang diambil dan kinerja retensi pelanggan. Gunakan umpan balik dari pelanggan untuk menyempurnakan strategi retensi.

bikin itungannya

**Simulasi:**
Misalnya, jika setiap minggu Anda menambahkan 200 pelanggan baru yang tidak melakukan churn, dalam dua minggu Anda akan menambah total 400 pelanggan baru. Dengan begitu, jumlah pelanggan yang tidak melakukan churn akan bertambah sebanyak 400 dalam dua minggu.

## ```REKOMENDASI```
#### **Segmentasi Pelanggan** 
Menggunakan informasi ini untuk melakukan segmentasi pelanggan berdasarkan karakteristik 

**Segmentasi Berdasarkan Usia:**

- Pelanggan Muda (18-30 tahun): Mereka mungkin lebih tertarik dengan fitur teknologi, promosi, dan diskon yang relevan dengan gaya hidup mereka. Strategi pemasaran yang fokus pada inovasi dan kenyamanan dapat menarik perhatian mereka.

- Pelanggan Dewasa (31-50 tahun): Fokus pada stabilitas keuangan, keamanan, dan manfaat jangka panjang bisa menjadi kunci. Rentang umur ini mungkin lebih tertarik dengan produk investasi atau tabungan dengan imbal hasil yang menarik.

- Pelanggan Tua (di atas 50 tahun): Prioritas mereka mungkin lebih pada kenyamanan, layanan pelanggan yang baik, dan fleksibilitas. Solusi yang disesuaikan dengan kebutuhan pensiun dan manfaat kesehatan bisa menjadi daya tarik.

**Segmentasi Berdasarkan Saldo Akun:**

- Pelanggan dengan Saldo Tinggi: Mereka dapat dianggap sebagai pelanggan VIP dan mungkin menikmati manfaat eksklusif, layanan prioritas, atau insentif lainnya untuk mempertahankan loyalitas mereka.

- Pelanggan dengan Saldo Rendah: Mereka mungkin lebih sensitif terhadap biaya dan kebijakan tarif. Strategi yang menawarkan diskon, penghematan, boundling produk atau program penghargaan bisa menjadi daya tarik.

**Segmentasi Berdasarkan Status Keanggotaan Aktif:**

- Pelanggan Aktif: Mereka mungkin lebih terbuka terhadap promosi dan insentif untuk memperluas partisipasi mereka dalam layanan. Strategi yang memperkuat keterlibatan mereka dan memberikan penghargaan atau reward atas setiap aktivitas yang dapat meningkatkan loyalitas.

- Pelanggan Tidak Aktif: Dapat fokus pada strategi untuk menghidupkan kembali minat mereka dalam layanan Anda. Ini bisa melibatkan penawaran khusus untuk memicu keterlibatan kembali, edukasi tentang manfaat layanan, atau perbaikan layanan pelanggan.

**Segmentasi Berdasarkan Gender:**

- Pelanggan Wanita: Mereka mungkin lebih responsif terhadap promosi yang menekankan kenyamanan, keamanan, atau perawatan diri. Strategi pemasaran yang menyoroti aspek-aspek ini bisa menarik minat mereka. contoh: bekerja sama dengan mitra kecantikan untuk metode pembayaran menggunakan Bank Berlian akan mendapatkan potongan harga, (diskon; 10% - 20%). Hari Valentine, Hari Ibu dan Perayaan lainnya.

- Pelanggan Laki-Laki: Mereka mungkin lebih tertarik pada promosi yang menonjolkan kepraktisan, keandalan, atau fitur teknologi. Strategi yang menekankan keunggulan teknologi atau efisiensi bisa lebih menarik bagi mereka. Contoh: Memberi diskon dengan metode pembayaran menggunakan Bank Berlian, menggunakan paket bundling pada produk.

**Segmentasi Berdasarkan Negara:**

- Pelanggan di Perancis: Mereka mungkin menanggapi promosi yang menekankan estetika, gaya hidup, atau budaya lokal. Strategi pemasaran yang menyesuaikan dengan preferensi budaya dan gaya hidup Perancis bisa menjadi kunci.

- Pelanggan di Jerman: Mereka mungkin lebih tertarik pada promosi yang menonjolkan kualitas, keandalan, atau prestasi. Strategi yang menekankan kehandalan produk atau layanan Anda bisa lebih efektif di pasar Jerman.

- Pelanggan di Spanyol: Mereka mungkin lebih responsif terhadap promosi yang menekankan kehangatan, kebersamaan, atau kesenangan. Strategi pemasaran yang menyoroti aspek-aspek ini bisa menarik minat mereka.

**Segmentasi Berdasarkan Skor Kredit:**

- Pelanggan dengan Skor Kredit Tinggi: Mereka mungkin lebih cenderung mencari manfaat jangka panjang, keamanan, dan pilihan premium. Strategi pemasaran yang menekankan manfaat jangka panjang atau layanan eksklusif bisa menarik bagi mereka. Karena skor kredit tinggi cenderung loyal dalam keuangan sehingga harus terus meningkatkan pelayanan, seperti memberikan servis terbaik, ini dapat diterapkan untuk memberi ucapan di hari perayaan maupun ulang tahun konsumen.

- Pelanggan dengan Skor Kredit Rendah: Mereka mungkin lebih sensitif terhadap biaya, promosi, atau penawaran khusus. Strategi pemasaran yang menawarkan insentif untuk memperbaiki skor kredit mereka atau program penghargaan untuk pengeluaran mereka bisa menjadi daya tarik. Konsumen ini biasanya sulit untuk mendapatkan kredit sehingga kita dapat membuat program untuk membantu konsumen untuk meningkatkan skor kredit dan memberikan pelayanan terbaik agar konsumen patuh terhadap ketentuan penggunaan produk seperti aturan angsuran kredit.
