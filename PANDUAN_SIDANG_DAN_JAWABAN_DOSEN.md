# PANDUAN SIDANG & CONTEKAN JAWABAN PERTANYAAN DOSEN PENGUJI
**Judul Tugas Akhir:** Analisis Perbandingan YOLOv12 dan MLP untuk Mendeteksi Postur Workout Berbasis MediaPipe  
**Program Studi:** Teknologi Rekayasa Elektronika, Politeknik Negeri Batam (2026)  

---

## 🎯 Strategi & Alur Presentasi Demo yang Sukses (10-15 Menit)

Saat Anda diminta mendemokan aplikasi di depan dosen penguji:

1. **Langkah 1: Jangan langsung push-up di lantai!**
   - Ruang sidang biasanya sempit dan sudut kamera laptop terbatas.
   - Buka file **`JALANKAN_DEMO_VIDEO.bat`** (cukup *double-click*).
   - Pilih menu **`[1] Push-up POSTUR BENAR`**.
   - Tunjukkan ke dosen: *"Bapak/Ibu, ini deteksi postur benar secara real-time pada ~30 FPS dengan status 'Form benar'."*
2. **Langkah 2: Tunjukkan Deteksi Postur Salah & Koreksinya (Tabel 4)**
   - Masuk ke menu **`[2] Push-up POSTUR SALAH`**.
   - Tunjukkan teks di layar: *"Ketika punggung melengkung atau pinggul turun, algoritma langsung mendeteksi deviasi sudut garis tubuh dan memberikan teks koreksi: 'Pinggul turun terhadap garis tubuh'."*
3. **Langkah 3: Tunjukkan Komparasi Dua Pipeline (YOLOv12 vs MLP)**
   - Saat video berputar, tekan tombol **`2`** di keyboard untuk beralih ke YOLOv12.
   - Tunjukkan ke dosen: *"Ini adalah pipeline kedua menggunakan YOLOv12n end-to-end langsung dari piksel citra."*
   - Tekan tombol **`1`** untuk kembali ke MediaPipe + MLP.
4. **Langkah 4: Tunjukkan Demo Webcam (Opsional / Jika Dosen Meminta)**
   - Buka file **`JALANKAN_DEMO_WEBCAM.bat`**.
   - Lakukan gerakan squat menghadap samping kamera.

---

## 💡 7 Pertanyaan Wajib Dosen Penguji & Rekomendasi Jawabannya

### Q1: "Apa perbedaan mendasar antara kedua pipeline yang Anda bandingkan?"
> **Jawaban:**  
> *"Pipeline 1 menggunakan pendekatan **dua tahap (Two-Stage Pose-Based)**: citra diekstraksi terlebih dahulu oleh MediaPipe Pose Landmarker menjadi 33 titik keypoint spasial 3D dan dihitung sudut anatomisnya (siku, lutut, kelurusan punggung), baru kemudian diklasifikasi oleh model Multi-Layer Perceptron (MLP).*  
> *Sedangkan Pipeline 2 menggunakan pendekatan **deteksi objek langsung (One-Stage Vision-Based)** dengan arsitektur YOLOv12n, di mana model langsung mendeteksi kotak pembatas tubuh (bounding box) dan mengklasifikasikan kelas postur dari fitur piksel citra tanpa proses kalkulasi sudut terpisah."*

---

### Q2: "Mengapa akurasi gerakan MediaPipe + MLP (82.8%) lebih tinggi daripada YOLOv12 (64.4%)?"
> **Jawaban:**  
> *"Ada dua alasan teknis utama:*  
> 1. *MediaPipe menormalisasi variasi latar belakang, warna pakaian, dan pencahayaan ruangan karena hanya mengekstraksi koordinat kerangka tubuh (skeleton). Model MLP hanya fokus mempelajari relasi sudut biomekanika.*  
> 2. *YOLOv12 harus mempelajari fitur visual piksel sekaligus lokalisasi koordinat kotak orang. Pada variasi pakaian dan latar belakang yang kompleks, detektor objek membutuhkan data citra yang jauh lebih besar agar tidak mengalami abstention/keraguan pada ambang batas konfidensi tinggi."*

---

### Q3: "Mengapa latensi MediaPipe + MLP jauh lebih cepat (~34 ms / ~29.2 FPS) dibandingkan YOLOv12 (~94 ms / ~10.5 FPS)?"
> **Jawaban:**  
> *"MediaPipe Pose Landmarker dioptimalkan dengan model BlazePose yang sangat ringan untuk perangkat CPU, dan model MLP hanya mengolah vektor 1D berisi sudut sendi. Sementara itu, YOLOv12n memproses matriks citra konvolusional multiskala (R-ELAN dan Attention) pada ukuran input 384x384 piksel. Tanpa akselerator GPU (CUDA), komputasi matriks YOLO pada CPU laptop membutuhkan waktu sekitar ~94 ms per frame."*

---

### Q4: "Bagaimana sistem Anda tahu kapan waktu yang tepat untuk menilai gerakan? Mengapa tidak menilai setiap frame?"
> **Jawaban:**  
> *"Sesuai perancangan sistem pada Bab 3 proposal, sistem menerapkan logika **Phase-Aware Evaluation (Deteksi Titik Ekstrem)**. Penilaian postur tidak dilakukan sembarang saat orang masih berdiri atau baru mulai bergerak, melainkan dikunci pada saat sendi mencapai titik fleksi maksimal (**bottom phase**, misal sudut siku <125° pada push-up atau sudut lutut pada squat) dan titik ekstensi penuh (**top phase**). Jika fase gerakan belum tercapai, sistem sengaja menahan diri (waiting_phase) untuk mencegah salah vonis."*

---

### Q5: "Mengapa gerakan Push-up dan Squat dinilai secara penuh, sedangkan Plank dan Mountain Climber difokuskan pada pengenalan gerakan?"
> **Jawaban (Jawaban Ilmiah Tingkat Tinggi - Sangat Disukai Penguji):**  
> *"Kami menerapkan prinsip **integritas riset dan pencegahan data leakage (kebocoran data)**. Untuk Push-up dan Squat, kami menggunakan dataset Mendeley Data yang memiliki rekaman multi-subjek independen (23 orang berbeda), sehingga model form benar/salah dapat diuji secara objektif pada subjek baru.*  
> *Sedangkan pada dataset publik Plank dan Mountain Climber yang tersedia, variasi subjeknya masih terbatas pada rekaman yang sama. Secara kaidah metodologi ilmiah, kami menolak mengklaim akurasi form yang overfit pada subjek yang sama, dan merekomendasikan penambahan dataset rekaman subjek independen sebagai saran penelitian selanjutnya."*

---

### Q6: "Bagaimana sistem mencegah teks koreksi di layar tidak berkedip-kedip (flickering)?"
> **Jawaban:**  
> *"Sistem menerapkan mekanisme **Debouncing & Cooldown Visual** pada modul antarmuka (FeedbackGuard). Ketika suatu kesalahan postur terdeteksi, instruksi koreksi akan ditahan di layar selama beberapa frame agar terbaca jelas oleh pengguna, sementara kalkulasi sudut di latar belakang tetap berjalan secara real-time tanpa jeda komputasi."*

---

### Q7: "Berapa batas toleransi sudut yang digunakan untuk menilai postur salah?"
> **Jawaban:**  
> *"Berdasarkan Tabel 3 & Tabel 4 proposal kami:*  
> - *Push-up: Punggung dianggap salah jika deviasi garis bahu–pinggul–tumit (`hip_line_offset`) melorot ke bawah (hips_low) atau naik melengkung (hips_high).*  
> - *Squat: Dinilai dari sudut batang tubuh terhadap garis vertikal (`torso_vertical`) saat posisi jongkok terdalam, di mana condong berlebihan ke depan diklasifikasikan sebagai `torso_forward`.*  
> - *Kedalaman gerak dinilai dari sudut siku (<125° untuk kedalaman push-up penuh)."*
