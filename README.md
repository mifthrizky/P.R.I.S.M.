# P.R.I.S.M. (Predictive Retrieval for Industrial Safety & Maintenance)

# Product Requirements Document (PRD)

## AI-Driven Industrial Maintenance & Compliance Agent

### 1. Executive Summary

Sistem _end-to-end_ yang bertindak sebagai agen pemeliharaan industri hibrida. Sistem ini memonitor aliran data sensor (IT/OT integration) untuk memprediksi anomali peralatan, dan secara otomatis mengekstrak dokumen manual serta Standar Operasional Prosedur (SOP) secara _offline_ untuk memberikan instruksi terstruktur kepada teknisi.

### 2. Arsitektur & Tech Stack

Sistem tidak membutuhkan perangkat keras baru, melainkan berfokus pada _software pipeline_ yang terintegrasi.

- **Predictive Model:** Python, Scikit-learn, XGBoost. (Disimpan dan diload menggunakan `joblib`).
- **Retrieval System (RAG):** FAISS IVF untuk pencarian vektor, digabung dengan BM25 untuk pencarian _keyword-based_. Model embedding menggunakan Multilingual-E5.
- **Inference LLM:** Qwen2.5 dijalankan secara lokal (via Ollama/vLLM) tanpa koneksi internet eksternal untuk menjaga privasi data pabrik.
- **Backend & Orchestration:** FastAPI untuk menangani _webhook_ data sensor, merutekan logika _retrieval_, dan memastikan keluaran LLM sesuai skema menggunakan Pydantic.
- **Frontend (Opsional untuk Visualisasi):** Next.js dipadukan dengan Tailwind CSS untuk merender dasbor _real-time_ dan tabel tindakan mitigasi.

### 3. Data Penelitian (Training & Knowledge Base)

Untuk melatih model dan mengisi _vector database_, gunakan sumber data publik.

**A. Data Latih Prediktif (AI4I 2020 Predictive Maintenance Dataset):**
Dataset ini mengandung 10.000 baris data operasional mesin dengan fitur-fitur numerik.

- **Fitur Input:**
  - `Type`: Kualitas mesin (L, M, H).
  - `Air temperature [K]`
  - `Process temperature [K]`
  - `Rotational speed [rpm]`
  - `Torque [Nm]`
  - `Tool wear [min]`
- **Target Label (Multiclass/Binary):**
  - `Machine failure` (0 atau 1)
  - Jenis kegagalan: `TWF` (Tool Wear Failure), `HDF` (Heat Dissipation Failure), `PWF` (Power Failure), `OSF` (Overstrain Failure).

**B. Data Teks (Knowledge Base RAG):**
Unduh 3-5 dokumen PDF publik untuk dipecah (_chunking_) dan dimasukkan ke dalam FAISS.

- Manual operator mesin CNC (misal: Haas Automation manual). **Perhatian: Hindari menggunakan spesifikasi dari merek tertentu yang dilarang penggunaannya dalam profil profesional.** Gunakan merek netral atau _open-source hardware_.
- Standar K3 / ISO 45001 versi ringkas terkait prosedur isolasi energi (_Lockout/Tagout - LOTO_).

### 4. Data Uji Coba (Testing / Inference Payloads)

Saat mempresentasikan proyek ini, jalankan simulasi menggunakan muatan JSON berikut ke _endpoint_ API-mu untuk mendemonstrasikan bagaimana sistem merespons dua kondisi berbeda.

**Uji Coba 1: Kondisi Normal (True Negative)**
Sistem memproses metrik, model memprediksi probabilitas kerusakan rendah. RAG dan LLM _tidak_ dipicu.

```json
{
  "sensor_id": "CNC-Milling-01",
  "timestamp": "2026-08-31T08:00:00Z",
  "metrics": {
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1551,
    "torque_nm": 42.8,
    "tool_wear_min": 21
  }
}
```

_Expected Output:_ `{"status": "HEALTHY", "failure_probability": 0.02}`

**Uji Coba 2: Kondisi Anomali (True Positive -> Trigger RAG)**
Kecepatan rotasi anjlok, torsi melonjak (indikasi _Power Failure_ / _Overstrain_).

```json
{
  "sensor_id": "CNC-Milling-01",
  "timestamp": "2026-08-31T08:15:00Z",
  "metrics": {
    "air_temperature_k": 304.5,
    "process_temperature_k": 313.2,
    "rotational_speed_rpm": 1201,
    "torque_nm": 71.2,
    "tool_wear_min": 215
  }
}
```

_Expected Output:_
Karena anomali menembus ambang batas, model mengidentifikasi `PWF/OSF`. FAISS mencari dokumen SOP _overheating/overstrain_, dan Qwen2.5 menghasilkan JSON mitigasi:

```json
{
  "status": "CRITICAL_WARNING",
  "predicted_failure": "Overstrain Failure (OSF)",
  "confidence_score": 0.89,
  "mitigation_plan": {
    "immediate_action": "Hentikan spindel utama segera dan putus aliran listrik sekunder (LOTO Protokol C).",
    "required_ppe": ["Sarung tangan tahan panas", "Kacamata safety"],
    "referenced_docs": ["SOP-Overstrain-Mitigation-Rev2.pdf (Halaman 14)"]
  }
}
```

### 5. Kriteria Penerimaan (Acceptance Criteria)

Jika sistemmu tidak memenuhi angka-angka ini, proyek ini belum selesai.

1.  **Akurasi ML:** Model klasifikasi wajib mencapai **ROC-AUC minimal 0.88** pada dataset evaluasi.
2.  **Validasi Output:** LLM harus menghasilkan keluaran JSON terstruktur (di-validasi oleh Pydantic) dengan _success rate_ 100% tanpa ada teks naratif acak yang merusak format.
3.  **Latensi:** Waktu dari masuknya JSON anomali hingga keluarnya JSON _mitigation plan_ dari LLM tidak boleh lebih dari 3 detik pada pengujian lokal.
