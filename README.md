# Fraud_Detection_model
# 📌 Overview

The Fraud Detection Prediction App is a machine learning project designed to detect potentially fraudulent financial transactions.
It leverages a trained ML pipeline and provides an interactive Streamlit web interface where users can input transaction details and instantly receive predictions on whether the transaction might be fraudulent.

# 🚀 Features

1.🧠 Machine Learning Model: Detects fraudulent transactions using a pre-trained model (fraud_detection_pipeline.pkl).
2.💻 Streamlit Web App: Simple and interactive UI for entering transaction details.
3.⚙️ Real-time Prediction: Instantly classifies a transaction as “Fraudulent” or “Not Fraudulent”.
4.📊 Notebook Exploration: Includes a Jupyter Notebook (main.ipynb) for data preprocessing, model training, and evaluation.

# 🧩 Project Structure
FraudDetection/
│
├── main.ipynb                  # Jupyter notebook with data analysis & model building
├── app.py                      # Streamlit web app for user interaction
├── fraud_detection_pipeline.pkl # Trained model (not uploaded here)
├── requirements.txt             # (Recommended) List of dependencies
└── README.md                    # Project documentation

# 🛠️ Installation & Setup
1. Clone or download this repository
   git clone https://github.com/yourusername/FraudDetection.git
   cd FraudDetection
2. Create and activate a virtual environment
   python -m venv myenv
   myenv\Scripts\activate   # On Windows 
   source myenv/bin/activate  # On Mac/Linux
3. Install dependencies
   pip install streamlit pandas scikit-learn joblib
4. Run the Streamlit app
   streamlit run app.py


# 🧮 How It Works

1.User inputs transaction details:
    Transaction type
    Amount
    Sender and receiver balances
2.The app sends this data to the pre-trained model.
3.The model predicts whether the transaction is fraudulent (1) or non-fraudulent (0).
4.The app displays:
    🔴 Fraud Alert for suspicious transactions
    🟢 Safe Transaction for normal ones


# 📓 Model Development

The Jupyter notebook (main.ipynb) includes:
1.Data loading and cleaning
2.Feature engineering
3.Handling class imbalance (e.g., SMOTE)
4.Model training (Random Forest / other ML model)
5.Evaluation metrics (Precision, Recall, F1-score, ROC-AUC)

# 📦 Example Input
| Transaction Type | Amount | Old Balance (Sender) | New Balance (Sender) | Old Balance (Receiver) | New Balance (Receiver) |
| ---------------- | ------ | -------------------- | -------------------- | ---------------------- | ---------------------- |
| TRANSFER         | 1000   | 5000                 | 4000                 | 1000                   | 2000                   |


# 🧠 Output

✅ “This transaction looks like it is not a fraud”
⚠️ “This transaction can be fraud”

# 🧰 Technologies Used

Python
Streamlit
Scikit-learn
Pandas
Joblib
Jupyter Notebook

# 💡 Future Enhancements

1.Add real-time data streaming integration.
2.Use deep learning models for higher accuracy.
3.Implement transaction visualization dashboards.


# 👩‍💻 Author

Muskan Jhala
🎓 B.Tech in Artificial Intelligence & Data Science

