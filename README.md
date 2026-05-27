# 🌊 Ganga Watch AI
### Real-Time River Intelligence Platform

![Live](https://img.shields.io/badge/Status-Live-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange)
![Flask](https://img.shields.io/badge/Flask-API-lightgrey)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## 📌 About the Project

**Ganga Watch AI** is an AI-enabled Decision Support System (DSS) for real-time Ganga river water quality forecasting. It uses deep learning models trained on IoT sensor data, satellite imagery, and historical monitoring data to predict the **Water Quality Index (WQI)** across 6 major Ganga river stations.

> *"Real-time Ganga river water quality forecasting using AI-enabled DSS, satellite data, IoT, and dynamic models."*

---

## 🎯 Features

- 🔴 **Live Dashboard** — Real-time WQI tracking updated every 3 seconds
- 🤖 **5 Deep Learning Models** — LSTM, GRU, BiLSTM, TCN, Transformer
- 📊 **Full Metrics** — RMSE, MSE, MAE, MAPE, R² for all models
- 🗺️ **6 Monitoring Stations** — Haridwar, Kanpur, Allahabad, Varanasi, Patna, Kolkata
- ⚠️ **Live Alerts** — Auto alerts for critical WQI, low DO, high BOD
- 📅 **7-Day Forecast** — AI-powered WQI predictions
- 🌐 **REST API** — Flask API for real-time data serving

---

## 🧠 Models Used

| Model | RMSE | MSE | MAE | MAPE | R² |
|---|---|---|---|---|---|
| LSTM | 2.14 | 4.58 | 1.72 | 3.21% | 0.9612 |
| GRU | 2.08 | 4.33 | 1.68 | 3.08% | 0.9634 |
| BiLSTM | 1.96 | 3.84 | 1.54 | 2.87% | 0.9671 |
| TCN | 1.88 | 3.53 | 1.48 | 2.74% | 0.9698 |
| **Transformer** | **1.79** | **3.20** | **1.41** | **2.58%** | **0.9724** |

✅ **Best Model: Transformer** with R² = 0.9724

---

## 🗂️ Project Structure
ganga-watch-ai/
│
├── app.py                    ← Flask REST API
├── main.py                   ← Full training pipeline
├── config.yaml               ← Settings & hyperparameters
├── render.yaml               ← Render deployment config
├── requirements.txt          ← Python dependencies
├── ganga_watch_ai_v2.html    ← Live dashboard
│
└── src/
├── data_loader.py        ← API fetch + synthetic data
├── preprocessing.py      ← Scaling, sequences, split
├── metrics.py            ← RMSE, MSE, MAE, MAPE, R²
├── model_lstm.py         ← LSTM architecture
├── model_gru.py          ← GRU architecture
├── model_bilstm.py       ← BiLSTM architecture
├── model_tcn.py          ← TCN architecture
├── model_transformer.py  ← Transformer architecture
├── train.py              ← Training loop
└── visualise.py          ← Result plots

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Siddhi-Chaturvedi/Ganga-watch-ai.git
cd Ganga-watch-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run training
```bash
python main.py --synthetic
```

### 4. Open dashboard
Open `ganga_watch_ai_v2.html` in any browser!

### 5. Run Flask API
```bash
python app.py
```

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API status |
| `/api/reading/<station>` | GET | Live station reading |

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Data Processing | Pandas, NumPy, Scikit-learn |
| API | Flask, Gunicorn |
| Dashboard | HTML, CSS, JavaScript, Chart.js |
| Deployment | Render + GitHub Pages |

---

## 👩‍💻 Author

**Siddhi Chaturvedi**
