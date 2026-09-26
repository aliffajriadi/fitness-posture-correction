# Fitness Posture Correction — Reviewer & Client Guide

Repository ini berisi prototipe sistem deteksi gerakan (*exercise recognition*) dan penilaian kebenaran form (*form posture assessment*) untuk 5 gerakan workout: **Push-up, Squat, Plank, Sit-up, dan Mountain Climber**.

Aplikasi ini dilengkapi dengan antarmuka grafis (OpenCV) untuk melakukan verifikasi, pelabelan, dan peninjauan (*human-in-the-loop review*) pada kandidat frame dataset sebelum model dilatih.

---

## 🚀 Panduan Memulai Cepat (Quick Start)

### 1. Persiapan Environment
Pastikan komputer Anda sudah terpasang **Python 3.10+**. Buka terminal PowerShell di folder proyek ini:

```powershell
# 1. Buat virtual environment lokal
python -m venv .venv

# 2. Aktifkan virtual environment
.venv\Scripts\Activate.ps1

# 3. Pasang dependensi yang diperlukan
pip install -r requirements-prototype.txt
```

> **Catatan:** Anda tidak perlu mengunduh dataset video mentah ukuran puluhan GB. Frame ekstraksi berkualitas tinggi yang siap direview sudah tersedia lengkap di folder `data/derived/frames/`.

---

## 🔍 Menjalankan Sesi Review

Jalankan perintah berikut di PowerShell untuk mulai meninjau frame yang diprioritaskan:

```powershell
.venv\Scripts\python.exe -m tools.review_prototype --reviewer "Nama Anda" --queue manifests/prototype_review_priority.csv
```

> Gantilah `"Nama Anda"` dengan nama Anda sebagai reviewer agar tercatat di metadata manifest.

Jika ingin mereview gerakan tertentu saja secara bertahap, Anda dapat menambahkan parameter `--exercise`:
```powershell
# Contoh khusus Push-up:
.venv\Scripts\python.exe -m tools.review_prototype --reviewer "Nama Anda" --queue manifests/prototype_review_priority.csv --exercise push_up
```

---

## ⌨️ Panduan Tombol Kontrol (Review Hotkeys)

Saat jendela review terbuka, pastikan jendela review aktif dan gunakan tombol keyboard berikut:

| Tombol | Fungsi | Keterangan |
| :---: | :--- | :--- |
| **`v`** | **Confirm Box** | Konfirmasi kotak orang (*bounding box*). Tekan lagi untuk membatalkan (*toggle*). |
| **`b`** | **Redraw Box** | Gambar ulang kotak orang secara manual dengan mouse jika kotak otomatis kurang presisi. Tekan Enter setelah memilih kotak. |
| **`e`** | **Exercise** | Pilih/ganti jenis gerakan (`push_up`, `squat`, `plank`, `sit_up`, `mountain_climber`). |
| **`f`** | **Form** | Pilih status form gerakan: `correct` (benar), `incorrect` (salah), atau kosong/*unrated*. |
| **`p`** | **Phase** | Pilih fase gerakan: `bottom`, `top`, `middle`, `static`, atau `transition`. |
| **`c`** | **Cue-Free** | Menandai frame bebas dari petunjuk jawaban, teks instruksi, watermark, atau spoiler visual lainnya. |
| **`g`** | **Geometry** | Menandai bahwa geometri sendi utama tampak jelas untuk pengukuran sudut. |
| **`a`** | **Accept** | **Terima & Simpan frame**. Menyimpan data review ke manifest dan lanjut ke frame berikutnya. |
| **`x`** | **Reject** | **Tolak frame**. Anda akan diminta memasukkan alasan penolakan di terminal (misal: blur, orang terpotong, salah gerakan). |
| **`n`** | **Next** | Lewati frame ini untuk saat ini tanpa menyimpan perubahan. |
| **`q`** / **`ESC`** | **Quit** | Keluar dari sesi review. Seluruh frame yang sudah di-accept/reject tersimpan aman. |

---

## 📐 Panduan Penentuan Fase Gerakan (`phase`)

Model membutuhkan kepastian fase agar evaluasi form hanya dilakukan pada posisi sendi yang tepat:

| Fase | Definisi & Posisi Tubuh | Contoh Penggunaan |
| :--- | :--- | :--- |
| **`static`** | Posisi statis / isometrik (menahan posisi tanpa repetisi naik-turun). | **Plank** (seluruh durasi tahan posisi lurus). |
| **`bottom`** | **Titik terendah / fleksi sendi maksimal / titik paling kritis penilaian postur.** | **Push-up**: Dada di bawah, siku menekuk (<125°).<br>**Squat**: Posisi jongkok terdalam, paha paralel lantai.<br>**Sit-up**: Punggung berbaring rata di matras sebelum naik.<br>**Mountain Climber**: Lutut ditarik maksimal mendekati dada. |
| **`top`** | **Titik tertinggi / ekstensi sendi penuh / posisi awal-akhir gerakan.** | **Push-up**: Lengan lurus menopang badan di atas.<br>**Squat**: Posisi berdiri tegak lurus.<br>**Sit-up**: Badan terangkat naik penuh mendekati lutut. |
| **`middle`** | Posisi di antara `top` dan `bottom` (sedang turun atau sedang naik). | Siku push-up atau lutut squat sedang menekuk setengah jalan. |
| **`transition`**| Momen pergantian kaki atau transisi dinamis bolak-balik. | **Mountain Climber** saat kedua kaki sedang bertukar posisi di tengah jalan. |

---

## 📋 Syarat Sebelum Menekan `a` (Accept)

Agar kualitas data latih model terjamin, sebuah frame hanya dapat diterima jika memenuhi prasyarat berikut:
1. Kotak orang terkonfirmasi (`Box reviewed: yes` — tekan **`v`**).
2. Frame bebas petunjuk jawaban / teks bocoran (`Cue free: yes` — tekan **`c`**).
3. Fase gerakan sudah ditentukan (bukan `unknown` — tekan **`p`**).
4. Gerakan sudah dipilih (tekan **`e`**).
5. Jika form dipilih **`incorrect`**, terminal akan meminta Anda mengetikkan alasan/jenis kesalahan yang terlihat (misalnya: `hips_high`, `hips_low`, `torso_forward`, `knees_inward`).

Jika ada syarat yang belum lengkap, baris notifikasi oranye di bawah jendela review (`>> Cannot accept: ...`) akan memberitahukan tombol apa yang perlu ditekan.

---

## 💾 Menyimpan & Menyerahkan Hasil Review

Setiap kali Anda menekan **`a` (accept)** atau **`x` (reject)**, hasil review langsung otomatis disimpan ke dalam file:
- **`manifests/prototype_samples.csv`**

Setelah Anda selesai melakukan review:
1. Tutup jendela review dengan menekan **`q`** atau tombol `ESC`.
2. Commit dan push perubahan file `manifests/prototype_samples.csv` ke repository GitHub ini, atau kirimkan file tersebut kembali ke tim pengembang untuk proses evaluasi dan pelatihan model.
