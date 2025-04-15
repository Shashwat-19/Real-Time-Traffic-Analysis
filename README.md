# 🚦 Real-Time Traffic Analysis (Bangalore)

This project uses **TomTom's Traffic API** to fetch **real-time traffic data** for key locations in Bangalore. It analyzes the traffic flow using machine learning (SVM classifier) and provides visualization through an interactive map and performance graphs.

---

## 📍 Features

- 📡 Fetches **live traffic data** using TomTom API
- ⚙️ Uses **SVM (Support Vector Machine)** to classify traffic as **Normal** or **Congested**
- 📊 Visualizes **actual vs predicted** traffic labels in a bar chart
- 🗺️ Displays traffic conditions on a **folium-based interactive map**
- 🧠 ML model built with **scikit-learn**

---

## 📌 Locations Monitored

- MG Road  
- Whitefield  
- Electronic City  
- Hebbal  
- Yelahanka  

---

## 🧪 How It Works

1. Requests real-time speed and confidence data from TomTom API
2. Processes the data into features:
   - `Current Speed`
   - `Free Flow Speed`
   - `Confidence`
3. Labels traffic as `Congested` or `Normal` using thresholds
4. Trains an SVM model and predicts on test data
5. Displays a confusion matrix, classification report, and bar chart
6. Generates an HTML map with color-coded markers (Green = Normal, Red = Congested)

---

## 🚀 Running in Google Colab

> ✅ All dependencies install automatically  
> ✅ Real-time results on execution

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 📁 Output Files

- `bangalore_traffic_map.html` – Live traffic map
- `bangalore_traffic_data.csv` – Processed dataset
- Inline graphs & metrics in Colab

---

## 🔐 API Key

You’ll need a **TomTom API key** to run the project.

1. Go to: [https://developer.tomtom.com](https://developer.tomtom.com)
2. Create a free account and generate a key
3. Replace the `TOMTOM_API_KEY` value in the code

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (SVM)
- Folium
- Matplotlib
- TomTom Traffic API

---

## 📬 Contact

Made by [Shashwat](https://github.com/Shashwat-19)  
Drop a ⭐ if you find this useful!

---

## 📄 License

MIT License. Feel free to fork and build upon this!
