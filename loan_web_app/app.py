from flask import Flask, render_template, request
from model import LoanModel

app = Flask(__name__)
loan_model = LoanModel()
loan_model.train()  # load/train model

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Collect form inputs
        income = float(request.form["applicant_income"])
        co_income = float(request.form["coapplicant_income"])
        loan_amount = float(request.form["loan_amount"])
        loan_term = float(request.form["loan_term"])
        credit_history = int(request.form["credit_history"])
        gender = int(request.form["gender"])
        married = int(request.form["married"])
        education = int(request.form["education"])
        self_employed = int(request.form["self_employed"])
        property_area = int(request.form["property_area"])
        age = int(request.form["age"])

        # ✅ Compute Debt-to-Income Ratio
        dti = loan_amount / (income + co_income) if (income + co_income) > 0 else 0

        # Prepare input data for prediction
        input_data = {
            "ApplicantIncome": income,
            "CoapplicantIncome": co_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Gender": gender,
            "Married": married,
            "Education": education,
            "Self_Employed": self_employed,
            "Property_Area": property_area,
            "Age": age,
            "DTI": dti   # NEW FEATURE
        }

        # Predict using LoanModel
        result, prob = loan_model.predict(input_data)

        # Friendly labels
        result_label = "✅ Loan Approved" if result == 1 else "❌ Loan Not Approved"
        credit_label = "Good" if credit_history == 1 else "Bad"
        gender_label = "Male" if gender == 1 else "Female"
        married_label = "Yes" if married == 1 else "No"
        education_label = "Graduate" if education == 1 else "Not Graduate"
        self_employed_label = "Yes" if self_employed == 1 else "No"
        property_label = ["Rural", "Semiurban", "Urban"][property_area]

        return render_template(
            "index.html",
            result=result,
            result_label=result_label,
            prob=prob,
            credit_label=credit_label,
            gender_label=gender_label,
            married_label=married_label,
            education_label=education_label,
            self_employed_label=self_employed_label,
            property_label=property_label,
            dti_value=round(dti, 3)   # Pass DTI to template
        )

    return render_template(
    "index.html",
    result=None,
    prob=None
)



if __name__ == "__main__":
    app.run(debug=True)
