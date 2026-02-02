import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

class LoanModel:
    def __init__(self):
        self.model = LogisticRegression()
        self.scaler = StandardScaler()
        self.encoders = {}  # used only during training

    def train(self, csv_path=None):
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(__file__), "loan_prediction.csv")

        # Read CSV (with encoding fix if needed)
        data = pd.read_csv(csv_path, encoding="latin1")
        data.ffill(inplace=True)

        # Encode categorical columns
        cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
        for col in cat_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            self.encoders[col] = le

        # ✅ Include DTI in training features
        X = data[['ApplicantIncome','CoapplicantIncome','LoanAmount',
                  'Loan_Amount_Term','Credit_History',
                  'Gender','Married','Education',
                  'Self_Employed','Property_Area','Age','DTI']]
        y = data['Loan_Status'].map({'Y': 1, 'N': 0})

        # Train/test split
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale and fit
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def predict(self, input_data: dict):
        """
        input_data: dict with numeric values already from form
        Returns: prediction (0 or 1), probability
        """
        df = pd.DataFrame([input_data])

        # ✅ Ensure prediction uses same feature order as training
        expected_features = ['ApplicantIncome','CoapplicantIncome','LoanAmount',
                             'Loan_Amount_Term','Credit_History',
                             'Gender','Married','Education',
                             'Self_Employed','Property_Area','Age','DTI']
        df = df[expected_features]

        X_scaled = self.scaler.transform(df)
        pred = self.model.predict(X_scaled)[0]
        prob = self.model.predict_proba(X_scaled)[0][1]
        return pred, prob
