import streamlit as st
import joblib
import numpy as np
import pandas as pd
# ---------------- LOAD MODEL ---------------- #

model = joblib.load("mental_health_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("label_encoders.pkl")
df = pd.read_csv("Student Mental health.csv")


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Student Mental Health Assessment",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main{
    background:#f5f7fb;
}

.title{
    text-align:center;
    font-size:42px;
    color:#1565C0;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
    margin-bottom:25px;
}

.card{
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 0px 15px rgba(0,0,0,0.1);
}

.stButton>button{
    width:100%;
    background:#1565C0;
    color:white;
    border:none;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#0D47A1;
}

</style>
""",unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown(
    "<div class='title'>🧠 Student Mental Health Assessment</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Your Mental Wellness Matters ❤️</div>",
    unsafe_allow_html=True
)

# ---------------- FORM ---------------- #

st.markdown("<div class='card'>",unsafe_allow_html=True)

st.header("👤 Personal Information")

col1,col2=st.columns(2)

with col1:

    name=st.text_input("Full Name")

    gender=st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    age=st.number_input(
        "Age",
        17,
        35
    )

    course=st.selectbox(
        "Course",
        [
            "BCS",
            "BIT",
            "Engineering",
            "KOE",
            "Kirkhs",
            "BENL",
            "CTS",
            "Biomedical Science",
            "Laws",
            "Islamic Education"
        ]
    )

with col2:

    year=st.selectbox(
        "Current Year",
        [
            "year 1",
            "year 2",
            "year 3",
            "year 4"
        ]
    )

    cgpa=st.number_input(
        "CGPA (10 Point Scale)",
        min_value=0.0,
        max_value=10.0,
        value=8.0,
        step=0.1
    )

    marital=st.selectbox(
        "Marital Status",
        ["No","Yes"]
    )

st.divider()

st.header("🩺 Mental Health Questions")

col3,col4=st.columns(2)

with col3:

    anxiety=st.selectbox(
        "Do you have Anxiety?",
        ["No","Yes"]
    )

with col4:

    panic=st.selectbox(
        "Do you have Panic Attack?",
        ["No","Yes"]
    )

treatment=st.selectbox(
    "Did you seek any specialist for treatment?",
    ["No","Yes"]
)

submit=st.button("🔍 Analyze Mental Health")

st.markdown("</div>",unsafe_allow_html=True)
st.divider()

st.header("📊 Mental Health Insights")

st.subheader("Depression Status Distribution")

depression_count = df["Do you have Depression?"].value_counts()

st.bar_chart(depression_count)


st.subheader("Anxiety Distribution")

anxiety_count = df["Do you have Anxiety?"].value_counts()

st.bar_chart(anxiety_count)



# ---------------- TEMP OUTPUT ---------------- #

if submit:

    # -------- Convert 10 Point CGPA to 4 Point GPA -------- #

    gpa = (cgpa / 10) * 4

    if gpa < 2.0:
        cgpa_range = "0 - 1.99"

    elif gpa < 2.5:
        cgpa_range = "2.00 - 2.49"

    elif gpa < 3.0:
        cgpa_range = "2.50 - 2.99"

    elif gpa < 3.5:
        cgpa_range = "3.00 - 3.49"

    else:
        cgpa_range = "3.50 - 4.00"

    # -------- Encode Inputs -------- #

    gender = encoders["Choose your gender"].transform([gender])[0]

    course = encoders["What is your course?"].transform([course])[0]

    year = encoders["Your current year of Study"].transform([year])[0]

    cgpa = encoders["What is your CGPA?"].transform([cgpa_range])[0]

    marital = encoders["Marital status"].transform([marital])[0]

    anxiety = encoders["Do you have Anxiety?"].transform([anxiety])[0]

    panic = encoders["Do you have Panic attack?"].transform([panic])[0]

    treatment = encoders["Did you seek any specialist for a treatment?"].transform([treatment])[0]

    data = np.array([[

        gender,
        age,
        course,
        year,
        cgpa,
        marital,
        anxiety,
        panic,
        treatment

    ]])

    # -------- Scale -------- #

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    # Probability
    probability = model.predict_proba(data)[0][1]

    if submit:
        st.divider()

        st.header("🧠 Mental Health Report")

# Student Name
        st.write(f"👤 Student Name: **{name}**")


# Risk Percentage
        risk = probability * 100


        col1, col2 = st.columns(2)

        with col1:
           st.metric(
        "Depression Risk",
        f"{risk:.2f}%"
        )

        with col2:

            if risk < 40:
               status = "Low Risk 🟢"

            elif risk < 70:
              status = "Moderate Risk 🟡"

            else:
              status = "High Risk 🔴"


            st.metric(
        "Risk Level",
        status
        )
         
    if prediction == 1:

       st.error("⚠ High Chance of Depression")

       st.subheader("💡 Recommendations")

       st.write("""
      • Maintain a regular sleep schedule 😴  
      • Try relaxation techniques like meditation 🧘  
    • Talk with friends, family, or trusted people 💬  
    • Consider consulting a mental health professional if needed 🩺  
    """)
    else:

      st.success("✅ No Depression Detected")

    st.subheader("🌱 Keep Maintaining Your Mental Wellness")

    st.write("""
    • Continue healthy daily habits ✅  
    • Maintain good sleep routine 😴  
    • Stay physically active 🏃  
    • Take breaks during studies 📚  
    """)
    # ---------------- AI SUPPORT CHAT ---------------- #

st.divider()

st.header("🤖 Mental Wellness Assistant")

user_question = st.text_input(
    "Ask anything about stress, study pressure, or mental wellness"
)


if user_question:

    responses = {
        "stress": "Try deep breathing, short breaks, and organize your tasks.",
        "sleep": "Maintain a regular sleep schedule and avoid screens before sleeping.",
        "exam": "Plan your study time, revise regularly, and don't overload yourself.",
        "default": "Take care of yourself. Talking with trusted people can help."
    }


    question = user_question.lower()


    if "stress" in question:
        answer = responses["stress"]

    elif "sleep" in question:
        answer = responses["sleep"]

    elif "exam" in question:
        answer = responses["exam"]

    else:
        answer = responses["default"]


    st.success(answer)