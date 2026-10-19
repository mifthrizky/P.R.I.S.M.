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

## Development Status

### Phase 1: Reproducible ML Foundation — Complete

The original notebook remains an exploratory record. Production-facing ML logic now lives in `src/prism/ml/` and is exposed through reproducible command-line tools.

### Phase 2: Calibrated Predictive Engine — Complete

The engine compares Random Forest and XGBoost with identical five-fold splits, selects the candidate with the strongest average precision, calibrates its probabilities, and learns a safety-oriented threshold on a separate validation set. A second multi-label model predicts TWF, HDF, PWF, and OSF.

Current validated engine:

- Selected binary model: calibrated Random Forest.
- Holdout ROC-AUC: `0.9604`.
- Holdout failure recall: `0.9118`.
- Holdout failure precision: `0.3010`.
- Learned operational threshold: `0.01326`.
- Normal demo probability: `0.00379` (`HEALTHY`).
- Anomaly demo probability: `0.73965` (`CRITICAL_WARNING`, primary failure `OSF`).
- Failure-type metrics are recorded separately in `artifacts/model_metadata.json`.

The low binary threshold is intentional: missing a real failure is treated as more costly than sending a machine to manual review. TWF remains the weakest subtype because AI4I contains very few positive TWF samples; its limitation is exposed in metadata rather than hidden behind aggregate accuracy.

### Project Structure

```text
configs/                  Reproducible training configuration
artifacts/                Versioned model bundle and readable metadata
scripts/                  Phase validation commands
src/prism/                Reusable package source
  ml/                     Features, training, evaluation, inference, persistence
tests/                    Unit and integration tests
01_model-training.ipynb   Exploratory notebook
```

### Local Setup

Python 3.12 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e . -r requirements-dev.txt
```

### Train the Predictive Engine

Training reads the raw AI4I CSV, creates deterministic train/validation/test splits, compares model candidates, calibrates the selected binary model, learns safety thresholds, trains the multi-label failure classifier, and writes a versioned bundle to `artifacts/`.

```powershell
.\.venv\Scripts\python.exe -m prism.ml.train --config configs/training.json
```

Generated files:

- `artifacts/model_bundle.joblib`: calibrated binary pipeline, failure-type pipeline, and metadata.
- `artifacts/model_metadata.json`: dataset hash, features, selected model, candidate comparison, thresholds, runtime versions, and per-label metrics.

Only load the joblib artifact from this trusted repository. Joblib/pickle artifacts can execute code when loaded and must not be accepted from untrusted sources.

### Evaluate and Run Demo Inference

```powershell
.\.venv\Scripts\python.exe -m prism.ml.evaluate --config configs/training.json
.\.venv\Scripts\python.exe -m prism.ml.inference --demo
```

To run the current predictive-engine gate:

```powershell
.\scripts\validate_phase2.ps1
```

The gate runs all automated tests, evaluates the saved bundle, and verifies that the two PRD payloads take the expected healthy and critical branches.
