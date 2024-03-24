# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from scipy.stats import norm
from datetime import datetime


# %% [markdown]
# # ``` PROBLEM BACKGROUND ```
# 
# Anda adalah seorang data analyst di sebuah Bank Berlian. 
# Departemen sales menemukan bahwa terdapat pelanggan yang tiba-tiba berhenti menggunakan layanan produk yang ada (churn 20%). 
# Ini menimbulkan kekhawatiran karena dapat berdampak negatif pada pendapatan dan reputasi perusahaan.
# Untuk mengatasi masalah ini, Anda diminta untuk melakukan analisis perilaku pelanggan (Churn). 
# Hasil analisis Anda akan digunakan untuk menentukan rekomendasi yang tepat bagi perusahaan dalam menghadapi tantangan ini.

# %% [markdown]
# ``` OBJEKTIF ```
# - Temukan segmentasi dengan tingkat churn paling tinggi.
# - Melihat tingkat hubungan faktor lain terhadap tingkat churn.
# - Berikan rekomendasi berdasarkan hasil analisis churn.
# 
# ``` TUJUAN ```
# - Tujuan bisnis untuk mengurangi churn sebesar 35%.

# %% [markdown]
# ## ``` IMPORT DATASET ```

# %%
import csv

# menampilkan dataset
df = pd.read_csv('Bank-Customer-Churn-Prediction.csv')

df.head(5)

# %%
# menampilkan informasi dataset
df.info()

# %% [markdown]
# ## ```Data Cleaning```

# %% [markdown]
# Data cleaning bertujuan untuk mendapatkan data yang akurat dan relevean sehingga meminimalisir risiko kesalahan dalam pengambilan keputusan.

# %%
# Menghilangkan missing null
df.isnull().sum()

# %% [markdown]
# Output di atas menunjukkan bahwa tidak ada data yagn kosong **(missing null)**

# %%
# mengecek duplikat dalam DataFrame
duplicate_rows = df[df.duplicated()]

# menampilkan hasil
print("Baris duplikat:")
print(duplicate_rows)
print("Jumlah baris duplikat:", duplicate_rows.shape[0])

# %% [markdown]
# Output di atas menunjukkan tidak ada duplikasi data di setiap kolom dataframe

# %%
# mengecek outlier dalam DataFrame

# menyiapkan quartile dari variabel yang akan dihitung
Q_1 = df[['credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card','active_member', 'estimated_salary']].quantile(0.25)
Q_3 = df[['credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card','active_member', 'estimated_salary']].quantile(0.75)

# perhitungan interquartile pada variabel
i_Q = Q_3 - Q_1

# menentukan batas bawah dan batas atas pada variabel yang dihitung
batas_bawah = Q_1 - 1.5 * i_Q
batas_atas = Q_3 + 1.5 * i_Q

# menampilkan hasil perhitungan quartile
print('Batas bawah:')
print(batas_bawah)
print('Batas atas:')
print(batas_atas)

# membuat perhitungan outlier pada setiap variabel
outlier = df[(df[['credit_score',  'age', 'tenure', 'balance', 'products_number', 'credit_card','active_member', 'estimated_salary']] < batas_bawah) | (df[['credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card','active_member', 'estimated_salary']] > batas_atas) ].any(axis = 1)


# %% [markdown]
# Out di atas menujukkan informasi mengenai hasil perhitungan outlier pada setiap variabel 

# %%
# membuat looping untuk memeriksa dan menghapus baris yang mengandung outlier
for index, row in df.iterrows():
    
    is_outlier = (row[['credit_score',
                         'age', 
                         'tenure', 
                         'balance', 
                         'products_number', 
                         'credit_card','active_member', 
                         'estimated_salary']] < batas_bawah) | (row[['credit_score',
                                                                     'age',
                                                                     'tenure', 
                                                                     'balance', 
                                                                     'products_number', 
                                                                     'credit_card',
                                                                     'active_member', 
                                                                     'estimated_salary']] > batas_atas)
    if is_outlier.any():

        df = df.drop(index)

df

# %% [markdown]
# Hasil menunjukkan bahwa data cleaning untuk baris yang memiliki value outlier akan dieliminasi untuk mendapatkan data yang akurat.

# %%
# menampilkan informasi dataset yang telah clean
df.info()

# %% [markdown]
# Output di atas menampilkan informasi setelah data cleaning dilakukan sehingga dataframe siap diolah untuk analisis lebih lanjut.

# %% [markdown]
# 
# # ```EXPLONATORY VARIABEL```

# %% [markdown]
# ## **1. segmentasi dengan tingkat churn paling tinggi**

# %% [markdown]
# ## `DESKRIPSI STATISTIK`

# %%
# deskripsi statistik seluruh data (overal)
df.describe()

# %%
# deskripsi statistik data churn yang telah difilter
(df[df['churn'] == 1]).describe()

# %% [markdown]
# - Dari statistik keseluruhan, dapat dilihat bahwa rata-rata usia pelanggan adalah 37 tahun, sementara rata-rata usia pelanggan yang churn adalah 43 tahun. *Ini menunjukkan bahwa pelanggan yang churn cenderung lebih tua daripada pelanggan secara umum.*
# 
# - Saldo rata-rata untuk pelanggan churn ($90,902.41) juga lebih tinggi daripada saldo rata-rata untuk keseluruhan pelanggan ($76,434.06), *menunjukkan bahwa pelanggan dengan saldo yang lebih tinggi cenderung lebih mungkin untuk churn.*
# 
# - Persentase anggota aktif juga lebih rendah di antara pelanggan churn (35.75%) dibandingkan dengan keseluruhan pelanggan (50.37%), *menunjukkan bahwa keterlibatan aktif dapat menjadi faktor yang mempengaruhi keputusan churn.*

# %%
# memvalidasi distribusi churn 
df['churn'].value_counts()

# %% [markdown]
# Terdapat 7,677 pelanggan yang tidak churn dan 1,964 pelanggan yang churn. Untuk menghitung persentase churn, kita dapat menggunakan rumus:
# 
# churn = (jumlah pelanggan yang churn / total jumlah pelanggan) * 100% 
# 
# disribusi atas menunjukkan bahwa sebasar `20,37%` pelanggan yang churn
# 
# 

# %% [markdown]
# ## ```PRE PROCESSING```

# %% [markdown]
# `Membuat rentang umur pelanggan`

# %% [markdown]
# Pembuatan rentang umur [bins] dapat membantu dalam memahami dan menganalisis data untuk simplikasi data dan visualisasi yang lebih mudah serta mengidentifikasi pola.

# %%
# menggunakan qcut() untuk melakukan binning dengan frekuensi yang sama
df['age_bins'] = pd.qcut(df['age'],
                          q=10, 
                          labels=False, 
                          duplicates='drop') + 1

# tampilkan ouput
df.head(5)

# %% [markdown]
# `Membuat rentang saldo pelanggan`

# %%
# menghitung frekuensi histogram
frequency, bins = np.histogram(df['balance'], 
                               bins=10)

# membuat DataFrame untuk menampung data histogram
histogram_data = pd.DataFrame({'Rentang Balance': [f'{bins[i]} - {bins[i+1]}' for i in range(len(bins)-1)],
                               'Frekuensi': frequency})

# menampilkan DataFrame
histogram_data


# %%
# menentukan batas-batas rentang
bins = [0, 
        25089, 
        50179, 
        75269, 
        100359, 
        125449, 
        150538, 
        175628, 
        200718, 
        250899]

# memberi label pada rentang
labels = ['0 - 25089', 
          '25089 - 50179', 
          '50179 - 75269', 
          '75269 - 100359', 
          '100359 - 125449', 
          '125449 - 150538', 
          '150538 - 175628', 
          '175628 - 200718', 
          '225808 - 250899']

# membuat kolom baru "balance_range" dengan menggunakan pd.cut() dan tipe data object
df['balance_range'] = pd.cut(df['balance'], 
                             bins=bins, 
                             labels=labels, 
                             include_lowest=True).astype(object)

# menampilkan output
df

# %%
# Buat dictionary untuk memetakan balance_range ke balance_bins
range_to_bins = {
    '0 - 25089': 1,
    '25089 - 50179': 2,
    '50179 - 75269': 3,
    '75269 - 100359': 4,
    '100359 - 125449': 5,
    '125449 - 150538': 6,
    '150538 - 175628': 7,
    '175628 - 200718': 8,
    '200718 - 250899': 9
}

# Menggunakan map untuk memetakan balance_range ke balance_bins
df['balance_bins'] = df['balance_range'].map(range_to_bins)

# Tampilkan DataFrame dengan kolom baru yang sudah disesuaikan
df


# %%
# menghitung frekuensi histogram
frequency, bins = np.histogram(df['credit_score'], 
                               bins=10)

# membuat DataFrame untuk menampung data histogram
histogram_data = pd.DataFrame({'Rentang credit score': [f'{bins[i]} - {bins[i+1]}' for i in range(len(bins)-1)],
                               'Frekuensi': frequency})

# menampilkan DataFrame
histogram_data

# %%
# menentukan batas-batas rentang
bins = [383.0, 429.7, 523.1, 569.8, 616.5, 663.2, 709.9, 756.6, 803.3, 850.0]

# memberi label pada rentang
labels = [
    '383.0 - 429.7',
    '429.7 - 523.1',
    '523.1 - 569.8',
    '569.8 - 616.5',
    '616.5 - 663.2',
    '663.2 - 709.9',
    '709.9 - 756.6',
    '756.6 - 803.3',
    '803.3 - 850.0'
]

# membuat kolom baru "credit_score_range" dengan menggunakan pd.cut() dan tipe data object
df['credit_score_range'] = pd.cut(df['credit_score'], bins=bins, labels=labels, include_lowest=True).astype(object)

# menampilkan output
df


# %%
churn_data = df[df['churn'] == 1]
churn_data 

# %% [markdown]
# ## ``UNVARITE ANALISYST``

# %% [markdown]
# ## `Visualisasi Data`
# visualisasi data membantu dapat melihat kompleksitas data, distribusi, pola dan tren data yang ada.

# %%
import matplotlib.pyplot as plt

# atur ukuran gambar
plt.figure(figsize=(10, 6))

# buat histogram rentang umur dengan anotasi nilai frekuensi
n, bins, patches = plt.hist(churn_data ['age'],
                            bins=10, 
                            color='skyblue', 
                            edgecolor='black', 
                            histtype='barstacked')

# judul plot
plt.title('Histogram Berdasarkan Rentang Umur Pelanggan')

# label sumbu x
plt.xlabel('umur')

# label sumbu y
plt.ylabel('frekuensi')

# tambahkan anotasi nilai frekuensi di atas setiap bar histogram
for i in range(len(patches)):
    plt.annotate(f'{int(n[i])}', 
                    xy=(patches[i].get_x() + patches[i].get_width() / 2, 
                     patches[i].get_height()), 
                     xytext=(0, 5), 
                     textcoords='offset points', 
                     ha='center', 
                     va='bottom')

# tampilkan plot
plt.show()


# %%
import matplotlib.pyplot as plt

# atur ukuran gambar
plt.figure(figsize=(10, 6))

# buat histogram rentang umur dengan anotasi nilai frekuensi
n, bins, patches = plt.hist(churn_data['balance'], 
                            bins=10, 
                            color='skyblue', 
                            edgecolor='black', 
                            histtype='barstacked')

# judul plot
plt.title('Histogram Berdasarkan Rentang Saldo Pelanggan')

# label sumbu x
plt.xlabel('balance')

# label sumbu y
plt.ylabel('frekuensi')

# tambahkan anotasi nilai frekuensi di atas setiap bar histogram
for i in range(len(patches)):
    plt.annotate(f'{int(n[i])}', 
                 xy=(patches[i].get_x() + patches[i].get_width() / 2, 
                 patches[i].get_height()), 
                 xytext=(0, 5), 
                 textcoords='offset points', 
                 ha='center', 
                 va='bottom')

# tampilkan plot
plt.show()


# %%
churn_data.info()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Mengatur ukuran dan layout subplot
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Visualisasi pertama: Countplot untuk jumlah churn berdasarkan tenure
sns.countplot(data=churn_data, x='tenure', hue='churn', palette='deep', ax=ax1)
ax1.set_title('Perbandingan Tenure Berdasarkan Churn')
ax1.set_xlabel('Tenure')
ax1.set_ylabel('Jumlah')
for p in ax1.patches:
    ax1.annotate(f'{p.get_height()}', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Visualisasi kedua: Barplot untuk presentase churn berdasarkan tenure
sns.barplot(data=df, x='tenure', y='churn', ci=None, estimator=lambda x: sum(x == 1) / len(x) * 100, color='skyblue', ax=ax2)
ax2.set_title('Presentase Churn Berdasarkan Tenure')
ax2.set_xlabel('Tenure')
ax2.set_ylabel('Presentase Churn (%)')
for p in ax2.patches:
    ax2.annotate(f'{p.get_height():.2f}%', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Menampilkan plot
plt.tight_layout()
plt.show()


# %% [markdown]
# Berdasarkan analisis data, terdapat variasi naik turunnya tingkat churn (berdasarkan persentase) seiring dengan kategori tenure. Namun, secara umum, terlihat bahwa tingkat churn cenderung tinggi pada tenure awal dan kemudian mengalami penurunan atau stabilisasi seiring dengan berjalannya waktu tenure.
# 
# - Pada awal tenure (tenure 0 hingga 1), tingkat churn relatif tinggi, mencapai sekitar 22-23%, menunjukkan bahwa pelanggan baru memiliki kemungkinan lebih tinggi untuk berhenti berlangganan.
# 
# - Seiring dengan meningkatnya tenure (tenure 2 hingga 4), tingkat churn mulai menurun dan cenderung stabil di sekitar 18-20%.
# 
# - Meskipun ada fluktuasi dalam tingkat churn di beberapa kategori tenure (seperti pada tenure 7), secara umum tidak terlihat tren peningkatan yang signifikan setelah melewati masa-masa awal.
# 
# Kesimpulannya, meskipun ada variasi di setiap kategori tenure, terlihat bahwa tingkat churn cenderung menurun atau stabil seiring dengan berjalannya waktu. Ini mungkin menunjukkan bahwa semakin lama pelanggan menggunakan layanan, semakin sedikit kemungkinan mereka akan berhenti berlangganan. 

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Mengatur ukuran dan layout subplot
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Visualisasi pertama: Countplot untuk jumlah churn berdasarkan jumlah produk
sns.countplot(data=df, x='products_number', hue='churn', palette='deep', ax=ax1)
ax1.set_title('Perbandingan Produk Nomor Berdasarkan Churn')
ax1.set_xlabel('Produk Nomor')
ax1.set_ylabel('Jumlah')
for p in ax1.patches:
    ax1.annotate(f'{p.get_height()}', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Visualisasi kedua: Barplot untuk presentase churn berdasarkan jumlah produk
sns.barplot(data=df, x='products_number', y='churn', ci=None, estimator=lambda x: sum(x == 1) / len(x) * 100, color='skyblue', ax=ax2)
ax2.set_title('Presentase Churn Berdasarkan Produk Nomor')
ax2.set_xlabel('Produk Nomor')
ax2.set_ylabel('Presentase Churn (%)')
for p in ax2.patches:
    ax2.annotate(f'{p.get_height():.2f}%', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Menampilkan plot
plt.tight_layout()
plt.show()



# %% [markdown]
# - Jumlah pelanggan pada produk satu adalah yang tertinggi, namun persentase churnnya juga tinggi, mencapai 27,71%. Ini mungkin menunjukkan bahwa pelanggan dengan satu produk cenderung lebih rentan terhadap churn.
# - Meskipun jumlah pelanggan dengan produk dua sedikit lebih rendah dari pelanggan dengan satu produk, persentase churnnya jauh lebih rendah, hanya sekitar 7.42%. Ini menunjukkan bahwa pelanggan dengan produk kedua cenderung lebih stabil.
# - Sedangkan jumlah pelanggan dengan produk ketiga jauh lebih sedikit, persentase churnnya sangat tinggi, mencapai 83.27%. Ini bisa menunjukkan bahwa pelanggan dengan produk tiga mungkin mengalami kesulitan atau tidak puas dengan layanan, sehingga lebih cenderung untuk churn.
# 
# Dari analisis ini, dapat disimpulkan bahwa semakin banyak produk yang dimiliki, semakin rendah kemungkinan churnnya, dengan pengecualian pelanggan yang memiliki tiga produk, yang memiliki tingkat churn yang sangat tinggi.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Mengatur ukuran dan layout subplot
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Visualisasi pertama: Countplot untuk jumlah churn berdasarkan kartu kredit
sns.countplot(data=df, x='credit_card', hue='churn', palette='deep', ax=ax1)
ax1.set_title('Perbandingan Kartu Kredit Berdasarkan Churn')
ax1.set_xlabel('Kartu Kredit')
ax1.set_ylabel('Jumlah')
for p in ax1.patches:
    ax1.annotate(f'{p.get_height()}', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Visualisasi kedua: Barplot untuk presentase churn berdasarkan kartu kredit
sns.barplot(data=df, x='credit_card', y='churn', ci=None, estimator=lambda x: sum(x == 1) / len(x) * 100, color='skyblue', ax=ax2)
ax2.set_title('Presentase Churn Berdasarkan Kartu Kredit')
ax2.set_xlabel('Kartu Kredit')
ax2.set_ylabel('Presentase Churn (%)')
for p in ax2.patches:
    ax2.annotate(f'{p.get_height():.2f}%', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Menampilkan plot
plt.tight_layout()
plt.show()


# %% [markdown]
# - Konsumen yang tidak mempunyai Kartu kredit memiliki jumlah churn sebanyak 571 dan jumlah non-churn sebanyak 2250.
# -  Konsumen yang mempunyai Kartu kredit memiliki jumlah  churn sebanyak 1320 dan jumlah non-churn sebanyak 5427.
# - Presentase churn lebih tinggi untuk yang tidak memiliki kartu kredit sebesar **20.24%** dibandingkan dengan yang  memiliki kartu kredit sebesar **19.56%**
# 
# Dapat dilihat bahwa meskipun jumlah churn lebih rendah untuk yang tidak memiliki kartu kredit, presentase churnnya sedikit lebih tinggi dibandingkan dengan yang memiliki kartu kredit. 

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Mengatur ukuran dan layout subplot
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Visualisasi pertama: Countplot untuk jumlah churn berdasarkan kartu kredit
sns.countplot(data=df, x='active_member', hue='churn', palette='deep', ax=ax1)
ax1.set_title('Perbandingan Member Aktif Berdasarkan Churn')
ax1.set_xlabel('Member Aktif')
ax1.set_ylabel('Jumlah')
for p in ax1.patches:
    ax1.annotate(f'{p.get_height()}', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Visualisasi kedua: Barplot untuk presentase churn berdasarkan kartu kredit
sns.barplot(data=df, x='active_member', y='churn', ci=None, estimator=lambda x: sum(x == 1) / len(x) * 100, color='skyblue', ax=ax2)
ax2.set_title('Perbandingan Member Aktif Berdasarkan Churn')
ax2.set_xlabel('Member Aktif')
ax2.set_ylabel('Presentase Churn (%)')
for p in ax2.patches:
    ax2.annotate(f'{p.get_height():.2f}%', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Menampilkan plot
plt.tight_layout()
plt.show()


# %% [markdown]
# Member Tidak Aktif (0) memiliki jumlah churn sebesar 25.58%, sedangkan jumlah non-churn (tidak churn) adalah 74.42%.
# Member Aktif (1) memiliki jumlah churn sebesar 14.03%, sedangkan jumlah non-churn (tidak churn) adalah 85.97%.
# 
#  kita bisa melihat bahwa persentase churn lebih rendah di antara pelanggan yang merupakan anggota aktif (14,60%) dibandingkan dengan yang bukan anggota aktif (26,20%). Ini menegaskan bahwa keaktifan dalam menggunakan layanan bank memiliki dampak positif pada retensi pelanggan.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Menghitung jumlah churn dan non-churn untuk setiap rentang umur
churn_count_by_age_bins = df.groupby('age_bins')['churn'].value_counts().unstack().fillna(0)

# Menghitung presentase churn untuk setiap rentang umur
churn_percentage_by_age_bins = churn_count_by_age_bins.div(churn_count_by_age_bins.sum(axis=1), axis=0) * 100

# Mengatur ukuran dan layout subplot
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Visualisasi pertama: Countplot untuk jumlah churn berdasarkan rentang umur
sns.countplot(data=churn_data, x='age_bins', hue='churn', palette='deep', ax=ax1)
ax1.set_title('Perbandingan Rentang Umur Berdasarkan Churn')
ax1.set_xlabel('Rentang Umur')
ax1.set_ylabel('Jumlah')
for p in ax1.patches:
    ax1.annotate(f'{p.get_height()}', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Visualisasi kedua: Barplot untuk presentase churn berdasarkan rentang umur
sns.barplot(data=churn_percentage_by_age_bins.reset_index(), x='age_bins', y=1, color='skyblue', ax=ax2)
ax2.set_title('Presentase Churn Berdasarkan Rentang Umur')
ax2.set_xlabel('Rentang Umur')
ax2.set_ylabel('Presentase Churn (%)')
for p in ax2.patches:
    ax2.annotate(f'{p.get_height():.2f}%', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Menampilkan plot
plt.tight_layout()
plt.show()



# %% [markdown]
# - Terdapat kecenderungan bahwa semakin tua rentang umur, semakin tinggi presentase churn. Rentang umur dengan presentase churn tertinggi terletak pada kategori umur yang lebih tua, yaitu pada kategori umur 9 dan 10.
# - Rentang umur dengan presentase churn terendah terletak pada kategori umur yang lebih muda, khususnya pada kategori umur 1 dan 2.
# 
# Rentang umur yang lebih muda cenderung memiliki presentase churn yang lebih rendah, mungkin karena kecenderungan untuk memiliki koneksi jangka panjang dengan penyedia layanan.
# Rentang umur yang lebih tua cenderung memiliki presentase churn yang lebih tinggi, mungkin karena perubahan kebutuhan atau preferensi pelanggan seiring bertambahnya usia.
# 
# Perlu adanya strategi pemasaran yang berbeda untuk masing-masing rentang umur. Misalnya, fokus pada mempertahankan pelanggan di rentang umur yang lebih tua dengan menawarkan layanan yang disesuaikan dengan kebutuhan mereka.

# %% [markdown]
# cek age dengan balance, apa jelek baru churn, tanggungan.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Menghitung jumlah churn dan non-churn untuk setiap rentang pendapatan
churn_count_by_balance_bins = df.groupby('balance_bins')['churn'].value_counts().unstack().fillna(0)

# Menghitung presentase churn untuk setiap rentang pendapatan
churn_percentage_by_balance_bins = churn_count_by_balance_bins.div(churn_count_by_balance_bins.sum(axis=1), axis=0) * 100

# Mengatur ukuran dan layout subplot
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Visualisasi pertama: Countplot untuk jumlah churn berdasarkan rentang pendapatan
sns.countplot(data=df, x='balance_bins', hue='churn', palette='deep', ax=ax1)
ax1.set_title('Perbandingan Rentang Pendapatan Berdasarkan Churn')
ax1.set_xlabel('Rentang Pendapatan')
ax1.set_ylabel('Jumlah')
for p in ax1.patches:
    ax1.annotate(f'{p.get_height()}', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Visualisasi kedua: Barplot untuk presentase churn berdasarkan rentang pendapatan
sns.barplot(data=churn_percentage_by_balance_bins.reset_index(), x='balance_bins', y=1, color='skyblue', ax=ax2)
ax2.set_title('Presentase Churn Berdasarkan Rentang Pendapatan')
ax2.set_xlabel('Rentang Pendapatan')
ax2.set_ylabel('Presentase Churn (%)')
for p in ax2.patches:
    ax2.annotate(f'{p.get_height():.2f}%', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Menampilkan plot
plt.tight_layout()
plt.show()


# %% [markdown]
# Terdapat kecenderungan bahwa semakin tinggi rentang pendapatan, semakin rendah presentase churn.
# Rentang pendapatan dengan presentase churn tertinggi terletak pada kategori pendapatan yang lebih rendah, yaitu pada kategori pendapatan 2.0.
# Rentang pendapatan dengan presentase churn terendah terletak pada kategori pendapatan yang lebih tinggi, khususnya pada kategori pendapatan 1.0.
# - Pelanggan dengan pendapatan yang lebih rendah cenderung memiliki presentase churn yang lebih tinggi. Hal ini mungkin disebabkan oleh keterbatasan keuangan yang membuat mereka lebih sensitif terhadap perubahan harga atau kualitas layanan.
# - Pelanggan dengan pendapatan yang lebih tinggi cenderung memiliki presentase churn yang lebih rendah. Mereka mungkin lebih mampu membayar layanan tambahan atau lebih puas dengan layanan yang mereka terima.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Mengatur ukuran dan layout subplot
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Visualisasi pertama: Countplot untuk jumlah churn berdasarkan kartu kredit
sns.countplot(data=churn_data, x='gender', hue='churn', palette='deep', ax=ax1)
ax1.set_title('Perbandingan Gender Berdasarkan Churn')
ax1.set_xlabel('Gender')
ax1.set_ylabel('Jumlah')
for p in ax1.patches:
    ax1.annotate(f'{p.get_height()}', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Visualisasi kedua: Barplot untuk presentase churn berdasarkan kartu kredit
sns.barplot(data=df, x='gender', y='churn', ci=None, estimator=lambda x: sum(x == 1) / len(x) * 100, color='skyblue', ax=ax2)
ax2.set_title('Presentase Churn Berdasarkan Kartu Kredit')
ax2.set_xlabel('Kartu Kredit')
ax2.set_ylabel('Presentase Churn (%)')
for p in ax2.patches:
    ax2.annotate(f'{p.get_height():.2f}%', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Menampilkan plot
plt.tight_layout()
plt.show()

# %% [markdown]
# ## **2. Melihat tingkat hubungan faktor lain terhadap tingkat churn**

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Menghitung jumlah churn dan non-churn untuk setiap rentang pendapatan dan rentang usia
age_bins_count_by_balance_bins = churn_data.groupby('balance_bins')['age_bins'].value_counts().unstack().fillna(0).astype(int)

# Menampilkan heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(age_bins_count_by_balance_bins, annot=True, fmt='d', cmap='coolwarm', linewidths=0.5)
plt.title('Jumlah Churn Berdasarkan Rentang Pendapatan dan Rentang Umur')
plt.xlabel('Rentang Umur')
plt.ylabel('Rentang Pendapatan')
plt.show()


# %% [markdown]
# - Rentang pendapatan 1, 5, dan 6 memiliki jumlah churn yang lebih tinggi dibandingkan dengan rentang pendapatan lainnya. Ini menunjukkan bahwa pelanggan dengan pendapatan dalam rentang tersebut cenderung lebih rentan untuk melakukan churn.
# 
# - Di sisi lain, rentang usia 8, 9, dan 10 memiliki kecenderungan churn yang lebih tinggi dari rentang usia lainnya. Hal ini menunjukkan bahwa pelanggan pada rentang usia tersebut lebih mungkin untuk melakukan churn.
# 
# - pola umum menunjukkan bahwa rentang usia yang lebih tinggi cenderung memiliki persentase churn yang lebih besar, terutama pada rentang pendapatan yang lebih tinggi. Pola ini menunjukkan bahwa faktor-faktor yang berbeda, seperti pendapatan dan usia, berkontribusi terhadap tingginya tingkat churn. 
# 

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Menghitung jumlah churn dan non-churn untuk setiap rentang pendapatan dan rentang usia
age_bins_count_by_credit_card = churn_data.groupby('balance_bins')['credit_card'].value_counts().unstack().fillna(0).astype(int)

# Menampilkan heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(age_bins_count_by_credit_card, annot=True, fmt='d', cmap='coolwarm', linewidths=0.5)
plt.title('Jumlah Churn Berdasarkan Rentang Pendapatan dan Kartu Kredit')
plt.xlabel('Kartu Kredit')
plt.ylabel('Rentang Umur')
plt.show()

# %% [markdown]
# - Pada rentang umur 1, 5, dan 6 memiliki jumlah churn paling tinggi. 
# - Secara umum terdapat pola bahwa semakin tinggi rentang umur semakin tinggi pula jumlah churn terutama untuk konsumen yang menggunakan kartu kredit

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Menghitung jumlah churn dan non-churn untuk setiap rentang pendapatan dan kartu kredit
credit_card_count_by_balance_bins = churn_data.groupby('balance_bins')['credit_card'].value_counts().unstack().fillna(0).astype(int)

# Menampilkan heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(credit_card_count_by_balance_bins, annot=True, fmt='d', cmap='coolwarm', linewidths=0.5)
plt.title('Jumlah Churn Berdasarkan Rentang Pendapatan dan Kartu Kredit')
plt.xlabel('Kartu Kredit')
plt.ylabel('Rentang Pendapatan')
plt.show()


# %% [markdown]
# - Konsumen yang mempunyai kartu kredit cenderung memiliki jumlah churn yang lebih tinggi daripada yang tidak memiliki kartu kredit.
# - Dengan demikian, terdapat kecenderungan bahwa rentang pendapatan yang lebih tinggi dan memiliki kartu kredit cenderung memiliki jumlah churn yang lebih tinggi.

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Menghitung jumlah churn dan non-churn untuk setiap rentang pendapatan dan rentang usia
country_by_gender = churn_data.groupby('country')['gender'].value_counts().unstack().fillna(0).astype(int)

# Menampilkan heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(country_by_gender, annot=True, fmt='d', cmap='coolwarm', linewidths=0.5)
plt.title('Jumlah Churn Berdasarkan Negara & Gender')
plt.xlabel('Gender')
plt.ylabel('Negara')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Menghitung jumlah churn dan non-churn untuk setiap rentang pendapatan dan rentang usia
credit_card_by_credit_score = churn_data.groupby('credit_score_range')['credit_card'].value_counts().unstack().fillna(0).astype(int)

# Menampilkan heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(credit_card_by_credit_score , annot=True, fmt='d', cmap='coolwarm', linewidths=0.5)
plt.title('Jumlah Churn Berdasarkan Rentang Skor Kredit')
plt.xlabel('Kartu Kredit')
plt.ylabel('Rentang Skor Kredit')
plt.show()



# %% [markdown]
# ngebatasin kredit diatas 200 itu udah chunr

# %% [markdown]
# Berdasarkan visualisasi di atas menunjukkan bahwa di setiap negara gender perempuan cenderung melakukan churn dibandingkan gender laki-laki.
# Sedangkan pada visualisasi kedua menggambarkan bahwa terdapat peningkatan jumlah churn dari rentang skor kredit 569.8 - 616.5 ke 616.5 - 663.2, namun kemudian mengalami penurunan di rentang skor kredit berikutnya. Jumlah churn cenderung meningkat seiring dengan peningkatan rentang skor kredit terutama pada pemilik kartu kredit. Kemudian, rentang skor kredit yang lebih tinggi cenderung memiliki jumlah churn yang lebih rendah.

# %% [markdown]
# **Pengaruh Rentang Pendapatan:**
# Rentang pendapatan 1, 5, dan 6 menunjukkan jumlah churn yang lebih tinggi.
# Hal ini menandakan bahwa pelanggan dengan pendapatan di rentang tersebut cenderung lebih rentan untuk melakukan churn.
# 
# **Pengaruh Rentang Usia:**
# Rentang usia 8, 9, dan 10 memiliki kecenderungan churn yang lebih tinggi.
# Ini menunjukkan bahwa pelanggan pada rentang usia tersebut lebih mungkin untuk melakukan churn.
# 
# **Polap Perilaku Umum:**
# Pola umum menunjukkan bahwa rentang usia yang lebih tinggi cenderung memiliki persentase churn yang lebih tinggi, terutama pada rentang pendapatan yang lebih tinggi.
# Rentang umur 1, 5, dan 6 memiliki jumlah churn paling tinggi, menunjukkan kecenderungan bahwa semakin tinggi rentang usia, semakin tinggi pula jumlah churn, terutama untuk konsumen yang menggunakan kartu kredit.
# 
# **Pengaruh Kartu Kredit:**
# Konsumen yang memiliki kartu kredit cenderung memiliki jumlah churn yang lebih tinggi daripada yang tidak memiliki kartu kredit.
# Terdapat kecenderungan bahwa rentang pendapatan yang lebih tinggi dan memiliki kartu kredit cenderung memiliki jumlah churn yang lebih tinggi.

# %% [markdown]
# # ```ANALYSIS DATA```

# %% [markdown]
# ## Odd, Hipotesis, Ordinary Least Squares (OLS), Signifikansi: R-Square

# %%
# Menghitung jumlah churn dan non-churn untuk setiap nilai 'age'
churn_count_by_age = df.groupby('age')['churn'].value_counts().unstack().fillna(0)

# Menghitung proporsi churn untuk setiap nilai 'age'
churn_percentage_by_age = churn_count_by_age.div(churn_count_by_age.sum(axis=1), axis=0)

# Menghitung odd ratio untuk 'age'
odd_ratio_age = churn_percentage_by_age[1] / churn_percentage_by_age[0]

# Urutkan secara descending
odd_ratio_age_sorted = odd_ratio_age.sort_values(ascending=False)

# Menghitung jumlah churn dan non-churn untuk setiap nilai 'balance'
churn_count_by_balance = df.groupby('balance')['churn'].value_counts().unstack().fillna(0)

# Menghitung proporsi churn untuk setiap nilai 'balance'
churn_percentage_by_balance = churn_count_by_balance.div(churn_count_by_balance.sum(axis=1), axis=0)

# Menghitung odd ratio untuk 'balance'
odd_ratio_balance = churn_percentage_by_balance[1] / churn_percentage_by_balance[0]

# Menghitung odd ratio untuk 'active_member'
odd_ratio_active_member = df.groupby('active_member')['churn'].value_counts().unstack().fillna(0).iloc[1] / df.groupby('active_member')['churn'].value_counts().unstack().fillna(0).iloc[0]

# Menghitung odd ratio untuk 'credit_card'
odd_ratio_credit_card = df.groupby('credit_card')['churn'].value_counts().unstack().fillna(0).iloc[1] / df.groupby('credit_card')['churn'].value_counts().unstack().fillna(0).iloc[0]

# Menampilkan odd ratio untuk semua variabel
print("Odd Ratio for Age:")
print(odd_ratio_age_sorted)
print()


# %% [markdown]
# Untuk rentang usia yang lebih muda (misalnya, 18 hingga 25 tahun), odd ratio cenderung rendah, yang menunjukkan bahwa konsumen dalam rentang usia ini memiliki kecenderungan yang lebih rendah untuk churn.
# Namun, odd ratio mulai meningkat secara signifikan saat usia konsumen mencapai sekitar 30 tahun dan seterusnya. Analisis ini menunjukkan bahwa usia konsumen memiliki pengaruh yang signifikan terhadap tingkat churn, dengan kecenderungan churn yang lebih tinggi terlihat pada konsumen yang lebih tua.

# %%
print("Odd Ratio for Balance:")
print(odd_ratio_balance)
print()

# %% [markdown]
# odd rasio sebesar 0,15 atau 15%. Terdapat nilai odd ratio yang sangat tinggi (infiniti) untuk beberapa rentang balance yang berarti rentang balance sangat tinggi cenderung tidak churn. Sebaliknya, terdapat juga nilai odd ratio yang sangat rendah memiliki kecenderungan tinggi untuk churn. Analisis ini menunjukkan bahwa saldo akun konsumen adalah faktor penting yang memengaruhi tingkat churn, dengan kecenderungan untuk churn cenderung lebih rendah pada konsumen dengan saldo yang lebih tinggi, dan sebaliknya. 

# %%
print("Odd Ratio for Active Member:")
print(odd_ratio_active_member)
print()


# %% [markdown]
# Polanya menunjukkan bahwa menjadi anggota yang aktif cenderung memiliki pengaruh yang signifikan dalam mengurangi kemungkinan churn. Konsumen yang aktif memiliki Odd Ratio yang lebih rendah, menunjukkan bahwa mereka memiliki kecenderungan yang lebih rendah untuk melakukan churn dibandingkan dengan konsumen yang tidak aktif. 

# %%

print("Odd Ratio for Credit Card:")
print(odd_ratio_credit_card)
print()


# %% [markdown]
# Untuk pelanggan yang memegang kartu kredit (1), Odd Ratio adalah sekitar 2.31.
# Sedangkan untuk pelanggan yang tidak memegang kartu kredit (0), Odd Ratio adalah sekitar 2.41.
# Hal ini menunjukkan bahwa ada sedikit kecenderungan bahwa pemegang kartu kredit memiliki peluang churn yang lebih rendah daripada yang tidak memegang kartu kredit, namun perbedaannya tidak signifikan. Dengan kata lain, memiliki atau tidak memiliki kartu kredit tidak terlalu memengaruhi peluang seorang pelanggan untuk churn.

# %% [markdown]
# ## `Hipotesis Analisis`

# %% [markdown]
# **Hipotesis 1: Usia Mempengaruhi Tingkat Churn**
# 
# Hipotesis: Terdapat hubungan positif antara usia pelanggan dengan tingkat churn, semakin tua usia pelanggan semakin tinggi kemungkinan mereka untuk churn.
# 
# Rationale: Pelanggan yang lebih tua mungkin memiliki kebutuhan atau preferensi yang berubah dari waktu ke waktu, membuat mereka lebih cenderung untuk mencari alternatif layanan.
# 
# **Hipotesis 2: Saldo Akun Mempengaruhi Tingkat Churn**
# 
# Hipotesis: Terdapat hubungan negatif antara saldo akun pelanggan dengan tingkat churn. Semakin tinggi saldo akun pelanggan semakin rendah kemungkinan mereka untuk churn.
# 
# Rationale: Pelanggan dengan saldo akun yang lebih tinggi mungkin merasa lebih terikat secara finansial atau emosional dengan bank, mengurangi kemungkinan mereka untuk mencari layanan dari penyedia lain.
# 
# **Hipotesis 3: Keterlibatan Aktif Mempengaruhi Tingkat Churn**
# 
# Hipotesis: Terdapat hubungan negatif antara keterlibatan aktif pelanggan dengan tingkat churn. Pelanggan yang aktif memiliki tingkat churn yang lebih rendah daripada yang tidak aktif.
# 
# Rationale: Pelanggan yang aktif mungkin merasa lebih terhubung dengan bank dan merasakan manfaat yang lebih besar dari layanan yang disediakan, mengurangi keinginan mereka untuk mencari alternatif lain. 

# %% [markdown]
#  ## ``` Melihat hubungan variabel dari corelasi```

# %% [markdown]
# **metode korelasi**
# 
# - pearson : hubungan linier antar variabel, distribusi normal, data interval.
# - kendal : mengukur kesaman dalam urutan peringakt, tanpa asumsi distribusi, data ordinal.
# - spearman : mengukur kesamaan dalam peringkat variabel, anpa asumsi distribusi, data ordinal/interval.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Memilih kolom numerik yang relevan
numeric_columns = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary']

# Menghitung korelasi antar kolom numerik
correlation_matrix = df[numeric_columns + ['churn']].corr()

# Membuat gambar (figure) baru dengan ukuran 12x8 inch
plt.figure(figsize=(12, 8))

# Membuat heatmap untuk visualisasi korelasi
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})

# Menampilkan judul plot
plt.title('Korelasi antara Variabel Numerik dan Churn')

# Menampilkan plot
plt.show()


# %% [markdown]
# Melihat korelasi positif antara variabe churn dengan age dan balance yakni sebesar 0,35 dan 0,12 yang artinya masuk dalam kategori korelasi lemah. Meskipun begitu dapat ditetapkan bahwa terdapat antara churn dengan age dan balance. sehingga dapat diartikan bahwa semakin tua usia pelanggan maka semakin tinggi peluang untuk mereka churn. Sedangkan aktif member memiliki korelasi negatif terhadap churn sebesar 0,15 sehingga dapat diartikan bahwa member tidak aktif cenderung memiliki churn tinggi dibandingkan member aktif.

# %% [markdown]
#  ## ``` Uji Hipotesis```

# %% [markdown]
# **Hipotesis 1 (H1)**: Terdapat hubungan positif antara usia pelanggan dengan tingkat churn, di mana semakin tua usia pelanggan, semakin tinggi kemungkinan mereka untuk churn.
# 
# **Hipotesis Alternatif (H1)**: Usia pelanggan berpengaruh signifikan terhadap tingkat churn.
# 
# **Hipotesis Nol (H0)**: Tidak ada hubungan antara usia pelanggan dengan tingkat churn, atau tidak ada perbedaan yang signifikan dalam tingkat churn berdasarkan usia pelanggan.
# 
# 
# **Hipotesis 2 (H2)**: Terdapat hubungan positif antara saldo akun pelanggan dengan tingkat churn, di mana semakin tinggi saldo akun pelanggan, semakin rendah kemungkinan mereka untuk churn.
# 
# **Hipotesis Alternatif (H2)**: Saldo akun pelanggan berpengaruh signifikan terhadap tingkat churn.
# 
# **Hipotesis Nol (H0)**: Tidak ada hubungan antara saldo akun pelanggan dengan tingkat churn, atau tidak ada perbedaan yang signifikan dalam tingkat churn berdasarkan saldo akun pelanggan.
# 
# 
# 
# **Hipotesis 3 (H3)**: Terdapat hubungan negatif antara keterlibatan aktif pelanggan dengan tingkat churn, di mana pelanggan yang aktif memiliki tingkat churn yang lebih rendah daripada yang tidak aktif.
# 
# **Hipotesis Alternatif (H3)**: Keterlibatan aktif pelanggan berpengaruh signifikan terhadap tingkat churn.
# 
# **Hipotesis Nol (H0)**: Tidak ada hubungan antara keterlibatan aktif pelanggan dengan tingkat churn, atau tidak ada perbedaan yang signifikan dalam tingkat churn berdasarkan keterlibatan aktif pelanggan.

# %% [markdown]
# # ```OLS```

# %%
import statsmodels.api as sm

# Menentukan variabel dependen (tingkat churn)
y = df['churn']

# Menentukan variabel independen (age dan balance)
X = df[['age', 'balance', 'active_member']]

# Menambahkan kolom konstan untuk model regresi
X = sm.add_constant(X)

# Membuat model regresi linier menggunakan OLS
ols_model = sm.OLS(y, X)

# Melakukan fitting model ke data
result = ols_model.fit()

# Menampilkan hasil summary
print(result.summary())


# %% [markdown]
# Koefisien (coef): Koefisien menunjukkan seberapa besar perubahan dalam variabel dependen (churn) yang diharapkan ketika variabel independen (age, balance, dan active_member) mengalami peningkatan satu satuan. Misalnya, koefisien untuk age adalah 0.0160, yang berarti bahwa ketika usia pelanggan naik satu tahun, kita harapkan peningkatan 0.0160 pada nilai churn.
# 
# P-value: P-value adalah ukuran signifikansi statistik dari model regresi secara keseluruhan. P-value yang sangat kecil (0.00) menunjukkan bahwa setidaknya satu variabel independen memiliki pengaruh signifikan terhadap variabel dependen.
# 
# R-squared: R-squared adalah proporsi variasi dalam variabel dependen yang dapat dijelaskan oleh variabel independen dalam model. Nilai R-squared sebesar 0.158 menunjukkan bahwa sekitar 15.8% variasi dalam tingkat churn dapat dijelaskan oleh variabel independen dalam model ini.
# 
# Kesimpulan: Berdasarkan hasil OLS ini, variabel independen (age, balance, dan active_member) secara bersama-sama memiliki pengaruh yang signifikan terhadap tingkat churn, dengan p-value yang sangat rendah. Oleh karena itu, hasil ini mendukung hipotesis bahwa ketiga variabel tersebut mempengaruhi tingkat churn.
# Berdasarkan hasil OLS Regression, hipotesis yang benar adalah:
# 
# - Hipotesis 1 (H1): Usia pelanggan (age) berpengaruh signifikan terhadap tingkat churn. Koefisien untuk usia adalah 0.0160 dengan p-value < 0.05, yang berarti ada hubungan positif antara usia pelanggan dan tingkat churn.
# 
# - Hipotesis 2 (H2): Saldo akun pelanggan (balance) juga berpengaruh signifikan terhadap tingkat churn. Koefisien untuk saldo akun adalah 6.369e-07 dengan p-value < 0.05, yang berarti ada hubungan negatif antara saldo akun pelanggan dan tingkat churn.
# 
# - Hipotesis 3 (H3): Keterlibatan aktif pelanggan (active_member) juga berpengaruh signifikan terhadap tingkat churn. Koefisien untuk keterlibatan aktif adalah -0.1201 dengan p-value < 0.05, yang berarti ada hubungan negatif antara keterlibatan aktif pelanggan dan tingkat churn.
# 
# 

# %%
import pandas as pd
from scipy.stats import chi2_contingency

# Membuat tabel kontingensi antara products_number dan churn
contingency_table = pd.crosstab(df['products_number'], df['churn'])

# Menampilkan tabel kontingensi
print("Tabel Kontingensi:")
print(contingency_table)

# Menghitung chi-square
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Menampilkan hasil
print("\nChi-square value:", chi2)
print("P-value:", p_value)
print("Degrees of freedom:", dof)
print("Expected frequencies table:")
print(expected)


# %% [markdown]
# 1. **Chi-square value**: Nilai chi-square yang dihitung adalah 1259.145.
# `products_number` dan `churn`.
#    - Semakin besar nilai chi-square, semakin kuat hubungannya.
# 
# 2. **P-value**: P-value yang dihasilkan sangat kecil (3.8027561084902225e-274), mendekati nol.
#    - Ini menunjukkan bahwa ada hubungan yang signifikan antara variabel `products_number` dan `churn`.
# 
# Dengan demikian, berdasarkan nilai chi-square yang tinggi dan p-value yang sangat rendah, dan ada hubungan yang signifikan antara keduanya. Variabel `products_number` dapat menjadi prediktor yang kuat untuk memprediksi perilaku churn pelanggan.

# %%
import pandas as pd
from scipy.stats import chi2_contingency

# Membuat tabel kontingensi antara products_number dan churn
contingency_table = pd.crosstab(df['active_member'], df['churn'])

# Menampilkan tabel kontingensi
print("Tabel Kontingensi:")
print(contingency_table)

# Menghitung chi-square
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Menampilkan hasil
print("\nChi-square value:", chi2)
print("P-value:", p_value)
print("Degrees of freedom:", dof)
print("Expected frequencies table:")
print(expected)


# %% [markdown]
# 1. **Chi-square value**: Nilai chi-square yang dihitung adalah 200.715.
#    - Semakin besar nilai chi-square, semakin kuat hubungannya.
# 
# 2. **P-value**: P-value yang dihasilkan sangat kecil (1.4583404225969284e-45), mendekati nol.
#    - Ini menunjukkan bahwa ada hubungan yang signifikan antara variabel `active_member` dan `churn`.
# 
# Dengan demikian, berdasarkan nilai chi-square yang tinggi dan p-value yang sangat rendah, ada hubungan yang signifikan antara keduanya. Variabel `active_member` dapat menjadi prediktor yang kuat untuk memprediksi perilaku churn pelanggan.

# %%
import pandas as pd
from scipy.stats import chi2_contingency

# Membuat tabel kontingensi antara products_number dan churn
contingency_table = pd.crosstab(df['credit_card'], df['churn'])

# Menampilkan tabel kontingensi
print("Tabel Kontingensi:")
print(contingency_table)

# Menghitung chi-square
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Menampilkan hasil
print("\nChi-square value:", chi2)
print("P-value:", p_value)
print("Degrees of freedom:", dof)
print("Expected frequencies table:")
print(expected)


# %% [markdown]
# 
# 
#  **Variabel `credit_card`**:
#    - Chi-square value yang diperoleh (0.5327) menunjukkan bahwa tidak ada hubungan yang signifikan antara variabel `credit_card` dan `churn`.
#    - P-value yang cukup besar (0.4655) menunjukkan bahwa tidak cukup bukti untuk menolak hipotesis nol, yang menyatakan bahwa tidak ada hubungan antara `credit_card` dan `churn`.
# berdasarkan hasil analisis ini, variabel `credit_card` tidak memiliki pengaruh yang signifikan terhadap kemungkinan churn pelanggan.

# %% [markdown]
# # **3. Rekomendasi berdasarkan hasil analisis churn**

# %% [markdown]
# ## ```RELEVANSI```

# %% [markdown]
# **Data Visualisasi:**
# - Terdapat pola bahwa semakin tinggi rentang umur dan rentang pendapatan maka semakin tinggi pula jumlah churn yang terjadi.  Kemudian,  Jumlah churn cenderung meningkat seiring dengan peningkatan rentang skor kredit terutama pada pemilik kartu kredit sedangkan rentang skor kredit yang lebih tinggi cenderung memiliki jumlah churn yang lebih rendah. Berdasarkan visualisasi  konsumen yang memiliki kartu kredit cenderung memiliki jumlah churn yang lebih tinggi daripada yang tidak memiliki kartu kredit. Pada setiap negara, gender perempuan cenderung melakukan churn lebih tinggi dibandingkan gender laki-laki.
# 
# **Korelasi:**
# - Age: 0.353756 (positif)
# - Balance: 0.115052 (positif)
# - Active Member: -0.145099 (negatif)
# - Products Number: -0.109205 (negatif)
# - Credit Card: -0.007750 (negatif)
# - Tenure: -0.014092 (negatif)
# - Korelasi tertinggi adalah antara usia (age) dan churn, menunjukkan hubungan positif yang cukup kuat.
# 
# **Uji Hipotesis:**
# 
# - Hipotesis 1 (H1): Terbukti bahwa usia pelanggan berpengaruh signifikan terhadap tingkat churn. Semakin tua usia pelanggan, semakin tinggi kemungkinan mereka untuk churn.
# - Hipotesis 2 (H2): Terbukti bahwa saldo akun pelanggan berpengaruh signifikan terhadap tingkat churn. Semakin tinggi saldo akun pelanggan, semakin rendah kemungkinan mereka untuk churn.
# - Hipotesis 3 (H3): Terbukti bahwa keterlibatan aktif pelanggan berpengaruh signifikan terhadap tingkat churn. Pelanggan yang aktif memiliki tingkat churn yang lebih rendah daripada yang tidak aktif.
# 
# 
# berdasarkan data visualisasi, korelasi, dan uji hipotesis, kita dapat menyimpulkan bahwa faktor-faktor seperti usia, saldo akun, dan keterlibatan aktif pelanggan memiliki pengaruh yang signifikan terhadap tingkat churn dalam dataset tersebut.

# %% [markdown]
# ## ```IN CASE TWO WEEKS```

# %%
# Jumlah total pelanggan yang churn dan tidak churn
churn_counts = {
    0: 7677,
    1: 1891
}

# Persentase menurunnya churn yang diinginkan
target_churn_rate = -0.35

# Jumlah total pelanggan
total_customers = sum(churn_counts.values())

# Jumlah pelanggan yang harus bertahan
target_retained_customers = int(total_customers * (1 - target_churn_rate))

# Jumlah pelanggan yang harus dipertahankan
customers_to_retain = target_retained_customers - churn_counts[1]

print("Jumlah pelanggan yang harus dipertahankan:", customers_to_retain)


# %% [markdown]
# ## ```SIMULATION```

# %% [markdown]
# **Relevansi:**
# Analisis churn memberikan wawasan tentang perilaku pelanggan dan faktor-faktor yang memengaruhi keputusan mereka untuk tetap menggunakan layanan atau beralih ke pesaing. Dengan pemahaman ini, perusahaan dapat mengambil langkah-langkah strategis untuk meningkatkan retensi pelanggan dan mengurangi churn, yang pada gilirannya dapat meningkatkan pendapatan dan keuntungan perusahaan.
# 
# **Rekomendasi:**
# 1. **Segmentasi Pelanggan:** Identifikasi kelompok pelanggan berdasarkan karakteristik seperti usia, saldo akun, dan keterlibatan aktif. Ini dapat membantu dalam menyesuaikan strategi retensi yang sesuai dengan kebutuhan dan preferensi masing-masing segmen.
# 2. **Program Loyalty:** Sertakan program loyalitas yang menarik untuk mendorong retensi pelanggan. Program seperti penghargaan, diskon, atau penawaran khusus dapat membantu meningkatkan keterlibatan pelanggan dan mengurangi kecenderungan mereka untuk beralih.
# 3. **Analisis Penggunaan Produk:** Tinjau penggunaan produk oleh pelanggan dan identifikasi pola penggunaan yang menunjukkan kecenderungan churn. Berikan dukungan atau layanan tambahan kepada pelanggan yang mungkin menghadapi masalah atau kebutuhan tambahan.
# 4. **Komunikasi Proaktif:** Komunikasikan secara proaktif dengan pelanggan yang menunjukkan tanda-tanda potensial untuk melakukan churn. Tawarkan solusi atau bantuan yang sesuai dengan kebutuhan mereka untuk mencegah churn.
# 5. **Optimalkan Layanan Pelanggan:** Pastikan layanan pelanggan yang responsif dan efektif. Tanggapi pertanyaan, keluhan, atau masalah pelanggan dengan cepat dan secara memuaskan untuk meningkatkan kepuasan pelanggan.
# 
# **Peningkatan Dalam Dua Minggu:**
# Berdasarkan jumlah pelanggan yang harus dipertahankan dan waktu yang tersedia (dua minggu), perlu dilakukan peningkatan sebesar [jumlah peningkatan per minggu] setiap minggu. Simulasikan langkah-langkah tertentu yang diambil untuk meningkatkan retensi pelanggan dalam dua minggu, seperti:
# 
# 1. **Kampanye Retensi:** Jalankan kampanye retensi yang ditargetkan kepada pelanggan yang memiliki risiko churn tinggi.
# 2. **Promosi Khusus:** Tawarkan promosi khusus atau diskon kepada pelanggan yang berpotensi untuk churn untuk mendorong mereka tetap menggunakan layanan.
# 3. **Kegiatan Cross-Selling:** Gunakan kesempatan untuk mengenalkan produk atau layanan baru kepada pelanggan yang sudah ada untuk meningkatkan keterlibatan mereka.
# 4. **Peningkatan Kualitas Layanan:** Tingkatkan kualitas layanan pelanggan dengan memberikan pelatihan tambahan kepada staf dan meningkatkan responsivitas terhadap masalah pelanggan.
# 5. **Feedback dan Evaluasi:** Lakukan evaluasi mingguan terhadap langkah-langkah yang diambil dan kinerja retensi pelanggan. Gunakan umpan balik dari pelanggan untuk menyempurnakan strategi retensi.
# 
# bikin itungannya
# 
# **Simulasi:**
# Misalnya, jika setiap minggu Anda menambahkan 200 pelanggan baru yang tidak melakukan churn, dalam dua minggu Anda akan menambah total 400 pelanggan baru. Dengan begitu, jumlah pelanggan yang tidak melakukan churn akan bertambah sebanyak 400 dalam dua minggu.

# %% [markdown]
# ## ```REKOMENDASI```

# %% [markdown]
# #### **Segmentasi Pelanggan** 
# Menggunakan informasi ini untuk melakukan segmentasi pelanggan berdasarkan karakteristik 
# 
# **Segmentasi Berdasarkan Usia:**
# 
# - Pelanggan Muda (18-30 tahun): Mereka mungkin lebih tertarik dengan fitur teknologi, promosi, dan diskon yang relevan dengan gaya hidup mereka. Strategi pemasaran yang fokus pada inovasi dan kenyamanan dapat menarik perhatian mereka.
# 
# - Pelanggan Dewasa (31-50 tahun): Fokus pada stabilitas keuangan, keamanan, dan manfaat jangka panjang bisa menjadi kunci. Rentang umur ini mungkin lebih tertarik dengan produk investasi atau tabungan dengan imbal hasil yang menarik.
# 
# - Pelanggan Tua (di atas 50 tahun): Prioritas mereka mungkin lebih pada kenyamanan, layanan pelanggan yang baik, dan fleksibilitas. Solusi yang disesuaikan dengan kebutuhan pensiun dan manfaat kesehatan bisa menjadi daya tarik.
# 
# **Segmentasi Berdasarkan Saldo Akun:**
# 
# - Pelanggan dengan Saldo Tinggi: Mereka dapat dianggap sebagai pelanggan VIP dan mungkin menikmati manfaat eksklusif, layanan prioritas, atau insentif lainnya untuk mempertahankan loyalitas mereka.
# 
# - Pelanggan dengan Saldo Rendah: Mereka mungkin lebih sensitif terhadap biaya dan kebijakan tarif. Strategi yang menawarkan diskon, penghematan, boundling produk atau program penghargaan bisa menjadi daya tarik.
# 
# **Segmentasi Berdasarkan Status Keanggotaan Aktif:**
# 
# - Pelanggan Aktif: Mereka mungkin lebih terbuka terhadap promosi dan insentif untuk memperluas partisipasi mereka dalam layanan. Strategi yang memperkuat keterlibatan mereka dan memberikan penghargaan atau reward atas setiap aktivitas yang dapat meningkatkan loyalitas.
# 
# - Pelanggan Tidak Aktif: Dapat fokus pada strategi untuk menghidupkan kembali minat mereka dalam layanan Anda. Ini bisa melibatkan penawaran khusus untuk memicu keterlibatan kembali, edukasi tentang manfaat layanan, atau perbaikan layanan pelanggan.
# 
# **Segmentasi Berdasarkan Gender:**
# 
# - Pelanggan Wanita: Mereka mungkin lebih responsif terhadap promosi yang menekankan kenyamanan, keamanan, atau perawatan diri. Strategi pemasaran yang menyoroti aspek-aspek ini bisa menarik minat mereka. contoh: bekerja sama dengan mitra kecantikan untuk metode pembayaran menggunakan Bank Berlian akan mendapatkan potongan harga, (diskon; 10% - 20%). Hari Valentine, Hari Ibu dan Perayaan lainnya.
# 
# - Pelanggan Laki-Laki: Mereka mungkin lebih tertarik pada promosi yang menonjolkan kepraktisan, keandalan, atau fitur teknologi. Strategi yang menekankan keunggulan teknologi atau efisiensi bisa lebih menarik bagi mereka. Contoh: Memberi diskon dengan metode pembayaran menggunakan Bank Berlian, menggunakan paket bundling pada produk.
# 
# **Segmentasi Berdasarkan Negara:**
# 
# - Pelanggan di Perancis: Mereka mungkin menanggapi promosi yang menekankan estetika, gaya hidup, atau budaya lokal. Strategi pemasaran yang menyesuaikan dengan preferensi budaya dan gaya hidup Perancis bisa menjadi kunci.
# 
# - Pelanggan di Jerman: Mereka mungkin lebih tertarik pada promosi yang menonjolkan kualitas, keandalan, atau prestasi. Strategi yang menekankan kehandalan produk atau layanan Anda bisa lebih efektif di pasar Jerman.
# 
# - Pelanggan di Spanyol: Mereka mungkin lebih responsif terhadap promosi yang menekankan kehangatan, kebersamaan, atau kesenangan. Strategi pemasaran yang menyoroti aspek-aspek ini bisa menarik minat mereka.
# 
# **Segmentasi Berdasarkan Skor Kredit:**
# 
# - Pelanggan dengan Skor Kredit Tinggi: Mereka mungkin lebih cenderung mencari manfaat jangka panjang, keamanan, dan pilihan premium. Strategi pemasaran yang menekankan manfaat jangka panjang atau layanan eksklusif bisa menarik bagi mereka. Karena skor kredit tinggi cenderung loyal dalam keuangan sehingga harus terus meningkatkan pelayanan, seperti memberikan servis terbaik, ini dapat diterapkan untuk memberi ucapan di hari perayaan maupun ulang tahun konsumen.
# 
# - Pelanggan dengan Skor Kredit Rendah: Mereka mungkin lebih sensitif terhadap biaya, promosi, atau penawaran khusus. Strategi pemasaran yang menawarkan insentif untuk memperbaiki skor kredit mereka atau program penghargaan untuk pengeluaran mereka bisa menjadi daya tarik. Konsumen ini biasanya sulit untuk mendapatkan kredit sehingga kita dapat membuat program untuk membantu konsumen untuk meningkatkan skor kredit dan memberikan pelayanan terbaik agar konsumen patuh terhadap ketentuan penggunaan produk seperti aturan angsuran kredit.


