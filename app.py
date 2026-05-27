from flask import Flask, jsonify
import numpy as np

app = Flask(__name__)

STATIONS = {
    "Haridwar":  {"wqi":72, "do":8.2, "ph":7.9, "bod":2.1, "tur":12, "tmp":18, "cond":320},
    "Kanpur":    {"wqi":41, "do":4.8, "ph":7.4, "bod":8.5, "tur":45, "tmp":24, "cond":580},
    "Allahabad": {"wqi":55, "do":6.1, "ph":7.6, "bod":5.2, "tur":28, "tmp":26, "cond":440},
    "Varanasi":  {"wqi":48, "do":5.4, "ph":7.5, "bod":6.8, "tur":36, "tmp":27, "cond":510},
    "Patna":     {"wqi":52, "do":5.8, "ph":7.6, "bod":5.8, "tur":32, "tmp":28, "cond":470},
    "Kolkata":   {"wqi":38, "do":4.2, "ph":7.3, "bod":9.8, "tur":58, "tmp":30, "cond":620},
}

def noise(amp):
    return float((np.random.random()-0.5)*2*amp)

@app.route("/api/reading/<station>")
def get_reading(station):
    if station not in STATIONS:
        return jsonify({"error": "Station not found"}), 404
    b = STATIONS[station]
    return jsonify({
        "station": station,
        "wqi":  round(max(0, min(100, b["wqi"]  + noise(4))),  2),
        "do":   round(max(0, min(14,  b["do"]   + noise(0.5))), 2),
        "ph":   round(max(6, min(9,   b["ph"]   + noise(0.15))),2),
        "bod":  round(max(0,          b["bod"]  + noise(1)),    2),
        "tur":  round(max(0,          b["tur"]  + noise(6)),    2),
        "tmp":  round(max(0,          b["tmp"]  + noise(1)),    2),
        "cond": round(max(0,          b["cond"] + noise(20)),   2),
    })

@app.route("/")
def home():
    return jsonify({"status": "Ganga Watch AI API running!"})

if __name__ == "__main__":
    app.run(debug=True)