# Laporan Proyek Machine Learning - Muhammad Raihan Fauzi

## Domain Proyek

**Latar Belakang**

Dalam beberapa tahun terakhir, perkembangan kecerdasan buatan (AI) meningkat pesat, terutama di bidang machine learning dan deep learning. NVIDIA (NVDA) memegang peranan penting sebagai penyedia GPU dan platform software yang menjadi tulang punggung pelatihan model AI modern. Ekspektasi pasar terhadap aplikasi AI—mulai dari kendaraan otonom, data center, hingga edge computing—membuat return saham NVDA berpotensi sangat tinggi, sekaligus menimbulkan volatilitas yang besar.

Namun, investasi memerlukan pertimbangan menyeluruh dari aspek fundamental, teknikal, dan makroekonomi. Prediksi time series pada proyek ini tidak dimaksudkan sebagai satu-satunya acuan, melainkan sebagai indikator tambahan dalam strategi investasi. Metode ini juga dapat diterapkan ulang untuk saham lain dengan cara melatih ulang model menggunakan data harga saham yang ingin dipreediksi.

**Mengapa Masalah Ini Penting?**
- Momentum AI: Adopsi AI di industri memicu lonjakan permintaan GPU dan harga saham NVDA.
- Keterbatasan Analisis Tradisional: Indikator teknikal sederhana belum menangkap pola kompleks yang diungkap model deep learning.
- Kebutuhan Otomatisasi: Sistem algoritmik trading & decision support memerlukan framework forecasting yang andal.

**Referensi**
- [Long Short-Term Memory Neural Network for Financial Time Series](https://arxiv.org/abs/2201.08218)
- [Optimizing Time Series Forecasting: A Comparative Study of Adam and Nesterov Accelerated Gradient on LSTM and GRU networks Using Stock Market data.](https://arxiv.org/abs/2410.01843)
- [Deep Learning for Time Series Forecasting](https://machinelearningmastery.com/deep-learning-for-time-series-forecasting/)
- [Time Series Analysis: Forecasting and Control](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118619193)

## Business Understanding

### Problem Statements
- Bagaimana memprediksi harga saham NVDA dalam beberapa waktu ke depan dengan ketepatan yang tinggi ?
- Bagaimana cara meningkatkan ketepatan prediksi harga saham dengan menggunakan machine learning ?
- Bagaimana cara mengetahui bahwa model yang di pilih merupakan model yang tepat untuk time series dan forecasting ? 

### Goals
- Membangun model machine learning dengan baik hingga dapat mencapai ketepatan yang tinggi untuk memprediksi harga saham NVDIA.
- Meningkatkan ketepatan model prediksi melalui beberapa hyperparameter tuning yang tepat.
- Membandingkan beberapa model machine learning yang diyakini dapat memberikan ketepatan prediksi yang tinggi.

### Solution Statement
- Multi-Model selection and comparison: Membandingkan beberapa model machine learning, pada proyek ini akan dibandingkan LSTM, GRU, dan CNN dengan metrik evaluasi Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE) serta akan dilakukan inferensi dengan cara membandingkan hasil testing model dengan actual price pada periode tertentu.
- Hyperparameter Tuning: Eksplorasi parameter—units (64, 32), dropout (0.2, 0.1), serta penggunaan early stopping.

## Data Understanding
Dataset yang digunakan berasal dari yahoo finance yang didownload langsung dari google colab menggunakan library yfinance [Yahoo Finance NVDA Historical](https://finance.yahoo.com/quote/NVDA/). Dataset ini berisi informasi tentang tanggal saham, harga open, high, low, close, dan volume perdagangan saham pada hari tertentu.

Periode Dataset : 2018‑01‑01 hingga 2025‑04‑24 (1.837 baris, 6 kolom).
| #   | Column         | Dtype      |
|-----|----------------|------------|
| 0   | Date           | datetime64 |
| 1   | Close          | float64    |
| 2   | High           | float64    |
| 3   | Low            | float64    |
| 4   | Open           | float64    |
| 5   | Volume         | int64      |

Gambaran data :
| Date       | Close    | High     | Low      | Open     | Volume    |
| ---------- | -------- | -------- | -------- | -------- | --------- |
| 2018-01-02 | 4.929428 | 4.933138 | 4.809500 | 4.841151 | 355616000 |
| 2018-01-03 | 5.253853 | 5.284268 | 5.038229 | 5.046884 | 914704000 |
| ...        | ...      | ...      | ...      | ...      | ...       |

Semua data valid: tak ada null, duplikat, atau outlier ekstrim.

### Variabel-variabel pada dataset saham NVIDA:
- Date = Tanggal pencatatan harga saham tersebut
- Close = Harga terakhir saham dihari tertentu karena bursa saham sudah tutup
- High = Harga tertinggi saham di hari tertentu
- Low = Harga saham terendah di hari tertentu
- Open = Harga buka saham NVDA ketika bursa saham buka di hari tertentu
- Volume = Volume saham yang diperdagangkan pada hari tertentu

**Exploratory Data Analysis (EDA)**

Tujuan EDA: Memahami distribusi, tren, pola, menangani data yang hilang, dan visualisasi data.

```python
print(df.shape)              # (1837, 6)
df.info()                    # Semua kolom non-null
df.duplicated().sum()        # 0 duplikat
df.isna().sum()              # 0 missing values
df.describe()                # Statistik deskriptif
```

- Visualisasi Tren Harga:
   ```python
   plt.figure(figsize=(14,7))
   plt.plot(df['Date'], df['Close'], label='Close')
   plt.title('Harga Penutupan NVDA')
   plt.xlabel('Tanggal'); plt.ylabel('Harga'); plt.legend(); plt.show()
   ```
   <img src="https://raw.githubusercontent.com/mraihanfauzii/PredictiveAnalytics/main/image/output.png" width="500">
   
   Memperlihatkan tren kenaikan tajam sejak 2023 dan level resistance di sekitar $150.

- Visualisasi Volume:
   ```python
   plt.figure(figsize=(14,5))
   plt.plot(df['Date'], df['Volume'], label='Volume', alpha=0.7)
   plt.title('Volume Perdagangan NVDA')
   plt.show()
   ```
   <img src="https://raw.githubusercontent.com/mraihanfauzii/PredictiveAnalytics/main/image/output1.png" width="500">
   
   Volume melonjak bersamaan dengan puncak harga, mengindikasikan aktivitas trading intens.

## Data Preparation
**Teknik Data Preparation**
- Scaling: MinMaxScaler pada kolom `Close`.
- Feature Engineering: Sliding window (lag 60 hari).
- Time-Based Split: 70% train, 15% validation, 15% test (chronological, shuffle=False).

**Proses Data Preparation**
```python
# Scaling
scaler = MinMaxScaler(); scaled = scaler.fit_transform(df[['Close']])
# Sliding window
X, y = create_dataset(scaled, time_steps=60)
# Chronological split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
```

**Alasan Tahapan Proses Data Preparation Dilakukan**
- Scaling: Menghindari perbedaan skala mempengaruhi training.
- Sliding Window: Menangkap pola jangka panjang dan seasonality.
- Chronological Split: Menjamin model diuji pada data future actual.
- Data Lengkap: Tak ada penghapusan karena tidak ada null/duplikat.

## Modeling
- Long Short-Term Memory (LSTM)

  Long Short-Term Memory (LSTM) adalah jenis Recurrent Neural Network (RNN) yang mampu mengingat dependensi jangka panjang melalui struktur sel memori dan mekanisme gate (input, forget, output).
  
  Arsitektur & Kode
  ```python
  model_lstm = Sequential([
    Input((60,1)),
    LSTM(64, return_sequences=True), Dropout(0.2),
    LSTM(32), Dropout(0.1),
    Dense(1)
  ])
  model_lstm.compile(Adam(1e-3), 'mse')
  ```
  - Input: Menetapkan format input (batch, timesteps, fitur).
  - LSTM(64, return_sequences=True): Layer pertama dengan 64 sel, return_sequences diperlukan agar layer kedua dapat menerima sequence lengkap.
  - Dropout(0.2): Mencegah dependency berlebih pada neuron; membantu generalisasi.
  - LSTM(32): Layer kedua lebih kecil (32 unit) untuk memfokuskan representasi ke fitur kunci.
  - Dropout(0.1): Regularisasi tambahan.
  - Dense(1): Layer output tunggal untuk memprediksi nilai harga.
  - Optimizer Adam & Loss MSE: Pilihan umum untuk regression time series.
  - Early Stopping: Pada proses fit, monitor val_loss agar training berhenti saat validasi tidak lagi membaik.

- Gated Recurrent Unit (GRU)
  Gated Recurrent Unit (GRU) adalah varian RNN yang menyederhanakan LSTM dengan dua gate (reset & update) sehingga lebih ringan dan cepat dilatih.

  Arsitektur & Kode
  ```python
  model_gru = Sequential([
    Input((60,1)),
    GRU(64, return_sequences=True), Dropout(0.2),
    GRU(32), Dropout(0.1),
    Dense(1)
  ])
  model_gru.compile(Adam(1e-3), 'mse')
  ```
  - Units (64, 32): Pertama menangkap pola jangka panjang, kedua mereduksi overfitting.
  - Dropout: Mencegah model terlalu menghafal.
  - Early Stopping: Digunakan pada .fit() untuk memantau val_loss dan menghentikan training bila tidak ada perbaikan.

- Convolutional Neural Network 1D
  Convolutional Neural Network 1D menerapkan convolution pada data sekuensial untuk menangkap pola lokal (motif) dalam window waktu.
  
  Arsitektur & Kode
  ```python
  model_cnn = Sequential([
    Input((60,1)),
    Conv1D(64,3,activation='relu'), MaxPooling1D(2),
    Conv1D(32,3,activation='relu'), MaxPooling1D(2),
    Flatten(), Dense(32,activation='relu'), Dropout(0.1),
    Dense(1)
  ])
  model_cnn.compile(Adam(1e-3), 'mse')
  ```
   - Conv1D + MaxPooling1D (dua kali): Tahap pertama menangkap pola harga lokal dan mereduksi noise; tahap kedua memperhalus dan mengurangi dimensi agar model fokus pada fitur paling dominan.
   - Flatten: Menyusun ulang tensor output menjadi vektor datar sebagai input ke Dense layer.
   - Dense(32) + Dropout(0.1): Dense menggabungkan fitur menjadi representasi global, Dropout mencegah model menghafal.
   - Dense(1): Menghasilkan prediksi harga tunggal.
   - Early Stopping: Pada proses fit, memantau val_loss untuk menghentikan training saat tidak ada peningkatan.

**Kelebihan & Kekurangan Setiap Algoritma**
- LSTM (Long Short-Term Memory)
  - Kelebihan :
      - Mengatasi Vanishing/Exploding Gradient : Struktur gate (input, forget, output) dan sel memori membuat LSTM sanggup mengingat informasi jangka panjang (long-term dependencies) tanpa gradien “menghilang” atau “meledak” selama backpropagation.
      - Fleksibilitas Sequence : Dapat menangani panjang sequence yang bervariasi—cocok untuk data time series di mana pola harian, mingguan, atau bulanan bisa berbeda.
      - Robust terhadap Noise : Gate forget membantu “melupakan” fluktuasi acak (noise) yang tidak relevan, menjaga model tetap fokus pada tren utama.
  - Kekurangan :
      - Kompleksitas Tinggi : Memiliki 4× lebih banyak parameter (gate dan sel) dibandingkan RNN standar, sehingga memerlukan memori dan komputasi lebih besar.
      - Waktu Latih & Inferensi Lebih Lama : Banyaknya lapisan dan parameter membuat epoch training lebih lambat, dan inferensi di aplikasi real-time bisa menjadi bottleneck.
      - Risiko Overfitting : Dengan parameter yang banyak, terutama di dataset terbatas, LSTM cenderung overfit jika tidak di-regularisasi (mis. dengan dropout atau early stopping).
      
- GRU (Gated Recurrent Unit)
  - Kelebihan :
      - Arsitektur Lebih Sederhana : Hanya punya dua gate (reset & update) dibanding empat gate di LSTM, sehingga parameter lebih sedikit (sekitar 25–33% lebih ringan).
      - Lebih Cepat & Efisien : Training per epoch lebih cepat dan memori yang digunakan lebih kecil—berguna saat sumber daya komputasi terbatas.
      - Performa Serupa di Banyak Kasus : Banyak studi (termasuk pada dataset saham) menunjukkan GRU mencapai akurasi sebanding dengan LSTM, meski lebih sederhana.
  - Kekurangan :
      - Kemampuan Ingatan Lebih Terbatas : Tanpa gate output eksplisit, GRU kadang kurang optimal dalam mempelajari very long-term dependencies (mis. pola musiman sepanjang bertahun-tahun).
      - Kurang Fleksibel : Struktur yang lebih ringkas terkadang tidak cukup menangkap kompleksitas pola yang sangat dinamis, sehingga LSTM bisa unggul saat dataset sangat besar dan rumit.
      - Sensitif terhadap Pengaturan Hyperparameter : Meski lebih sedikit parameter, pemilihan jumlah unit dan dropout yang tidak tepat bisa membuat performa turun drastis.
      
- CNN 1D (Convolutional Neural Network)
  - Kelebihan :
      - Deteksi Pola Lokal : Conv1D menangkap “motif” dalam window sempit (mis., pergerakan harga 3–5 hari) dengan kernel yang digeser, efektif mengenali sinyal teknikal jangka pendek.
      - Parallelisme & Efisiensi : Konvolusi bisa diparalelkan di GPU, sehingga latih jauh lebih cepat daripada RNN yang bersifat sekuensial.
      - Parameter Relatif Sedikit : Dengan pooling (MaxPooling1D) mengurangi dimensi tanpa kehilangan pola penting, jumlah parameter tetap terkendali.
  - Kekurangan :
      - Konteks Jangka Panjang Terbatas : Receptive field (total window yang “dilihat” jaringan) hanya sejauh beberapa kernel + pooling layer; untuk konteks >60–100 hari perlu stacking banyak lapisan atau teknik dilation.
      - Kurang Dinamis pada Sequence : CNN tidak memodelkan urutan waktu secara intrinsik—setiap window dianggap independen—sehingga kurang menangkap ketergantungan sekuensial dibanding RNN.
      - Desain Arsitektur Lebih Manual : Memilih ukuran kernel, jumlah lapisan, dan pooling perlu eksperimen tersendiri; tidak ada mekanisme gate yang otomatis menyesuaikan “berapa lama” ingatan disimpan.

**Solusi Model Terbaik**

Model terbaik yang telah dilatih akan dipilih dengan melihat metrik evaluasinya, model terbaik akan dipilih dengan MAE dan RMSE terkecil dari model lainnya. Jadi bila LSTM/GRU memiliki nilai MAE dan RMSE yang paling kecil bila dibandingkan model lainnya maka model tersebut menjadi model terbaik dalam proyek ini.

## Evaluation

**Metrik Evaluasi**
Dalam time series forecasting saham, metrik evaluasi harus mudah diinterpretasi dalam satuan asli (dolar) dan menunjukkan besaran kesalahan umum (MAE) dan menyoroti outlier (RMSE).

Mean Absolute Error (MAE), MAE adalah rata-rata nilai absolut selisih antara prediksi dan aktual yang berfungsi untuk memberikan gambaran rata-rata kesalahan, mudah diinterpretasi. Semakin kecil nilai MAE maka model semakin baik.

$$
\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^n \bigl|y_i - \hat y_i\bigr|
$$

- $y_i$: nilai aktual ke-\(i\)
- $\hat y_i$: nilai prediksi ke-\(i\)
- $n$: jumlah sampel

Root Mean Squared Error (RMSE), RMSE menyoroti kesalahan yang lebih besar secara kuadrat, memberi penalti lebih tinggi pada outlier. semakin kecil, semakin baik.

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat y_i)^2}
$$

- $y_i$: nilai aktual ke-\(i\)
- $\hat y_i$: nilai prediksi ke-\(i\)
- $n$: jumlah sampel

**Hasil Evaluasi**
| Model | MAE     | RMSE    |
| ----- | ------- | ------- |
| LSTM  | 10.7397 | 12.6856 |
| GRU   | 3.8096  | 4.9507  |
| CNN   | 14.0640 | 16.1201 |

Kriteria MAE:
- MAE < 5: sangat baik
- 5 ≤ MAE < 10: baik
- MAE ≥ 10: perlu perbaikan

Kriteria RMSE :
- RMSE < 10: sangat baik
- 10 ≤ RMSE < 20: baik
- RMSE ≥ 20: perlu perbaikan

**Inferensi & Model Terbaik**

Inferensi (prediksi vs aktual) :

```python
plt.figure(figsize=(12,5))
plt.plot(y_test_act, label="Actual", color="black")
plt.plot(inv_scale(model_lstm.predict(X_test_rnn)), label="LSTM")
plt.plot(inv_scale(model_gru.predict(X_test_rnn)),  label="GRU")
plt.plot(inv_scale(model_cnn.predict(X_test_rnn)),  label="CNN")
plt.title(f"{ticker} – Prediksi vs Aktual (Test Set)")
plt.xlabel("Hari")
plt.ylabel("Harga ($)")
plt.legend()
plt.show()
```
<img src="https://raw.githubusercontent.com/mraihanfauzii/PredictiveAnalytics/main/image/output2.png" width="500">

Grafik menampilkan prediksi LSTM, GRU, dan CNN vs harga aktual. GRU mengikuti pola aktual paling rapat.

**Model Terbaik**

GRU terpilih sebagai model terbaik: Berdasarkan metrik evaluasi MAE & RMSE memiliki nilai yang paling rendah hal ini terbukti ketika inferensi GRU mengikuti pola aktual paling tepat bila dibandingkan dengan LSTM dan CNN, bila dibandingkan pada nilai metrik evaluasi LSTM dan CNN memiliki nilai yang tidak begitu jauh hal tersebut terbukti juga pada inferensi yang mana LSTM dan CNN memiliki prediksi harga yang hampir mirip.

**Evaluasi terhadap Business Goals**
- Menjawab problem statement, mencapai goals, dan berhasil mencapai solution statement: Berhasilnya ditemukan model yang dapat memprediksi harga saham NVDA dengan ketepatan yang tinggi yang dibuktikan dengan nilai metrik evaluasi yang cukup bagus pada MAE dan RMSE serta model tersebut yaitu model GRU merupakan yang terbaik bila dibandingkan dengan LSTM dan CNN pada proyek ini. Penggunaan Hyperparameter Tuning yaitu parameter—units (64, 32), dropout (0.2, 0.1), serta penggunaan early stopping terbukti efektif pada proyek ini.
- Dampak Solution Statement: Proyek ini dapat menjadi indikator tambahan bagi investor untuk menganalisa saham NVDA berkat ketepatan prediksi yang tinggi.

## Kesimpulan

Setelah melalui keseluruhan alur—mulai dari definisi domain, business understanding, data understanding & EDA, data preparation, modeling, hingga evaluasi—terbukti bahwa arsitektur GRU paling unggul untuk forecasting harga saham NVDA. Metrik MAE=3.81 dan RMSE=4.95 memenuhi standar risiko yang ditetapkan. Meskipun LSTM dan CNN juga dapat memprediksi, GRU memberikan keseimbangan terbaik antara kecepatan pelatihan dan ketepatan prediksi. Proses hyperparameter tuning (unit, dropout, early stopping) terbukti efektif meningkatkan performa. Rangkaian langkah ini siap diulang untuk saham lain dengan hanya mengganti dataset historisnya.

Investasi yang bijak membutuhkan analisis menyeluruh—aspek fundamental perusahaan, indikator teknikal, hingga tren makroekonomi. Oleh karena itu, hasil time series forecasting dalam proyek ini hanya menjadi indikator tambahan, bukan satu-satunya dasar keputusan investasi. Tahapan-tahapan yang telah dilakukan pada proyek ini untuk memprediksi harga saham NVDIA dapat diterapkan ulang ke saham lain dengan melatih ulang model menggunakan data historis saham yang ingin diprediksi.

---
