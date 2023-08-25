import streamlit as st
import pandas as pd
import xgboost as xgb
# model = TabularPredictor.load("AutogluonModels/ag-20230825_174102/",require_py_version_match=False)

# model = pickle.load(open("AutogluonModels/ag-20230825_174102/predictor.pkl", "rb"))

model = xgb.XGBClassifier()
model.load_model('xgboost.json')

@st.cache_data
def predict(Age, Subscription_Length_Months, Monthly_Bill, Total_Usage_GB, Gender_Male):
    if Gender_Male=="Male":
        Gender_Male=1
    else:
        Gender_Male = 0
    usage_to_bill = Total_Usage_GB/Monthly_Bill
    usage_subs = Total_Usage_GB/Subscription_Length_Months
    usage_age = Total_Usage_GB/Age
    relative_bill = Monthly_Bill/65.05319680000001
    cost_per_month = Monthly_Bill / Subscription_Length_Months
    diff = Total_Usage_GB - Monthly_Bill
    input = pd.DataFrame(
            [
                [Age, Subscription_Length_Months, Monthly_Bill, Total_Usage_GB, Gender_Male, usage_to_bill, usage_subs ,usage_age, relative_bill, cost_per_month, diff]
                ], 
            columns=
                ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB','Gender_Male', 'usage_to_bill', 'usage_subs', 'usage_age', 'relative_bill', 'cost_per_month', 'diff']
            )
    op = model.predict(
            input
        )
    
    return op
    
predict(62, 1, 48.76, 172, 0)

st.title("Customer Churn Prediction")
st.header('Enter the characteristics of user:')

Age = st.number_input('Enter the Age')
Subscription_Length_Months = st.number_input('Enter the Subscription Length in Months')
Monthly_Bill= st.number_input('Enter the Monthly Bill')
Total_Usage_GB = st.number_input('Enter the Total Usage in GB')
Gender_Male = st.selectbox('Choose your gender', ['Male',"Female"])

if st.button('Predict churn'):
    churn = predict(Age, Subscription_Length_Months, Monthly_Bill, Total_Usage_GB, Gender_Male)
    st.success(f'The prediction that the customer will leave is {churn[0]}')