import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="MindCare | Student Wellness",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# LOAD EXISTING PROJECT FILES
# Keep these files in the same folder as app.py:
# mental_health_model.pkl, scaler.pkl,
# label_encoders.pkl, Student Mental health.csv
# -------------------------------------------------
@st.cache_resource
def load_model_files():
    model = joblib.load("mental_health_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return model, scaler, encoders

@st.cache_data
def load_dataset():
    return pd.read_csv("Student Mental health.csv")

try:
    model, scaler, encoders = load_model_files()
    df = load_dataset()
except Exception as error:
    st.error("Project files could not be loaded.")
    st.info(
        "Check that mental_health_model.pkl, scaler.pkl, "
        "label_encoders.pkl and Student Mental health.csv "
        "are present in the same folder as app.py."
    )
    st.caption(f"Technical details: {error}")
    st.stop()

# -------------------------------------------------
# CUSTOM DESIGN
# -------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #f7f5ff 0%, #f3f8ff 55%, #ffffff 100%);
        color: #30265f !important;
    }

    /* Force readable dark text when the Streamlit account theme is dark. */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] li,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stCaptionContainer"] {
        color: #39335f !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #39335f !important;
    }

    /* High-contrast info banner: readable text on a soft blue-lavender surface. */
    [data-testid="stAlert"] {
        background: linear-gradient(100deg, #e9e4ff 0%, #e3f1ff 100%) !important;
        border: 1px solid #c9d7ff !important;
        border-left: 5px solid #7561c9 !important;
        border-radius: 14px !important;
    }
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] div,
    [data-testid="stAlert"] span {
        color: #25345f !important;
        font-weight: 600 !important;
    }

    /* Give navigation controls a matching lavender accent. */
    [data-testid="stSidebar"] input[type="radio"] {
        accent-color: #7561c9 !important;
    }

    [data-testid="stSidebar"] {
        background: #f0edff;
        border-right: 1px solid #e5defd;
    }

    .hero {
        padding: 28px 30px;
        border-radius: 22px;
        background: linear-gradient(115deg, #e8e1ff 0%, #e3efff 100%);
        border: 1px solid #ded7fb;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: #30265f;
        font-size: 2.25rem;
        font-weight: 800;
        margin: 0 0 8px 0;
    }

    .hero p {
        color: #5f5b7b;
        font-size: 1.02rem;
        margin: 0;
    }

    .section-heading {
        color: #352b68;
        font-weight: 750;
        margin-top: 8px;
    }

    .soft-card {
        background: rgba(255,255,255,0.94);
        border: 1px solid #e1dcf7;
        border-top: 4px solid #8b78d6;
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: 0 5px 18px rgba(66, 54, 120, 0.05);
        margin-bottom: 14px;
    }

    .small-note {
        color: #77738e;
        font-size: 0.88rem;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #7561c9, #5688d9);
        color: white;
        border: 0;
        border-radius: 12px;
        min-height: 46px;
        font-weight: 700;
        width: 100%;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(90deg, #624eb5, #4175c7);
        color: white;
        border: 0;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e8e4f7;
        padding: 16px;
        border-radius: 16px;
    }

    h1, h2, h3 {
        color: #352b68;
    }

    .footer {
        text-align: center;
        color: #88839e;
        padding: 18px 0 8px 0;
        font-size: 0.85rem;
    }

    /* Consistent readable text for form controls and dark-theme overrides. */
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] {
        color: #30265f !important;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] div,
    [data-testid="stSelectSlider"] div {
        color: #30265f !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: #cfc7ed !important;
    }
    [data-testid="stForm"] {
        background: rgba(255,255,255,0.72);
        border: 1px solid #e3def5;
        border-radius: 18px;
        padding: 18px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 MindCare")
    st.caption("Student Mental Wellness Companion")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        ["Home", "Assessment", "Insights", "Wellness Assistant", "About"],
        label_visibility="visible"
    )
    st.divider()
    st.caption("A student-focused awareness and prediction project.")
    st.caption("This tool is not a medical diagnosis.")

# -------------------------------------------------
# HOME
# -------------------------------------------------
if page == "Home":
    st.markdown("""
    <div class="hero">
        <h1>Welcome to MindCare 🌿</h1>
        <p>A supportive space to reflect on student mental wellness.
        Understand the project, complete the assessment, and explore the dataset insights.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Your well-being matters")
    st.write(
        "Academic pressure, sleep changes and everyday challenges can affect "
        "how students feel. MindCare provides a simple assessment experience "
        "and general wellness suggestions."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="soft-card"><h3>📝 Assessment</h3><p>Complete the student information and mental-health questionnaire.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="soft-card"><h3>📊 Insights</h3><p>Explore the depression and anxiety distributions in the project dataset.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="soft-card"><h3>💬 Wellness support</h3><p>Get simple general suggestions related to stress, sleep and exams.</p></div>', unsafe_allow_html=True)

    st.info("Start with **Assessment** from the sidebar when you are ready.")

# -------------------------------------------------
# ASSESSMENT
# -------------------------------------------------
elif page == "Assessment":
    st.markdown("""
    <div class="hero">
        <h1>Student Wellness Check-in 🌿</h1>
        <p>A private moment to reflect on how you have been feeling lately.
        Answer honestly; there are no right or wrong answers.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "This check-in offers general self-care ideas. It does not diagnose a "
        "condition or calculate a clinical mental-health risk. Your reflection "
        "answers are separate from the original machine-learning model inputs."
    )

    frequency_options = [
        "Never", "A few days", "More than a week", "Nearly every day"
    ]
    support_options = [
        "Yes, definitely", "Sometimes", "Not at the moment"
    ]

    with st.form("mental_health_assessment"):
        st.markdown("### 👤 Student information")
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Name (optional)")
            gender_input = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=17, max_value=35, value=20)
            course_input = st.selectbox(
                "Course",
                [
                    "BCS", "BIT", "Engineering", "KOE", "Kirkhs",
                    "BENL", "CTS", "Biomedical Science", "Laws",
                    "Islamic Education"
                ]
            )

        with col2:
            year_input = st.selectbox(
                "Current Year", ["year 1", "year 2", "year 3", "year 4"]
            )
            cgpa_input = st.number_input(
                "CGPA (10 Point Scale)", min_value=0.0,
                max_value=10.0, value=8.0, step=0.1
            )
            marital_input = st.selectbox("Marital Status", ["No", "Yes"])

        st.divider()
        st.markdown("### 🧠 Existing project questionnaire")
        st.caption(
            "These are the original questionnaire fields used by the trained "
            "model. They are retained to preserve the project's ML pipeline."
        )
        col3, col4 = st.columns(2)
        with col3:
            anxiety_input = st.selectbox("Do you have Anxiety?", ["No", "Yes"])
        with col4:
            panic_input = st.selectbox("Do you have Panic Attack?", ["No", "Yes"])
        treatment_input = st.selectbox(
            "Did you seek any specialist for treatment?", ["No", "Yes"]
        )

        st.divider()
        st.markdown("### 💜 Your wellness reflection")
        st.caption(
            "Think about the last two weeks. These additional questions are "
            "used only to tailor the general suggestions shown below; they "
            "are not inputs to the pre-trained model."
        )

        stress = st.select_slider(
            "How often have you felt stressed or overwhelmed?",
            options=frequency_options, value="A few days"
        )
        low_mood = st.select_slider(
            "How often have you felt low, sad or emotionally drained?",
            options=frequency_options, value="A few days"
        )
        focus = st.select_slider(
            "How often has it been difficult to concentrate on your studies or daily tasks?",
            options=frequency_options, value="A few days"
        )
        sleep = st.select_slider(
            "How has your sleep been?",
            options=["Very restful", "Mostly okay", "Often disturbed", "Very difficult"],
            value="Mostly okay"
        )
        energy = st.select_slider(
            "How often have you felt too tired to do your usual activities?",
            options=frequency_options, value="A few days"
        )
        support = st.selectbox(
            "Do you have someone you feel comfortable talking to?",
            support_options
        )
        enjoyable = st.selectbox(
            "Have you made time for an activity you enjoy recently?",
            ["Yes, regularly", "Sometimes", "Hardly / not yet"]
        )

        submitted = st.form_submit_button("🌱 Create My Wellness Summary")

    if submitted:
        try:
            # Preserve the original trained model's preprocessing pipeline.
            # The prediction is intentionally not displayed as a diagnosis,
            # probability, or risk category. Wellness suggestions below are
            # based only on the separate reflection answers.
            gpa = (cgpa_input / 10) * 4
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

            gender_encoded = encoders["Choose your gender"].transform([gender_input])[0]
            course_encoded = encoders["What is your course?"].transform([course_input])[0]
            year_encoded = encoders["Your current year of Study"].transform([year_input])[0]
            cgpa_encoded = encoders["What is your CGPA?"].transform([cgpa_range])[0]
            marital_encoded = encoders["Marital status"].transform([marital_input])[0]
            anxiety_encoded = encoders["Do you have Anxiety?"].transform([anxiety_input])[0]
            panic_encoded = encoders["Do you have Panic attack?"].transform([panic_input])[0]
            treatment_encoded = encoders[
                "Did you seek any specialist for a treatment?"
            ].transform([treatment_input])[0]

            input_data = np.array([[
                gender_encoded, age, course_encoded, year_encoded,
                cgpa_encoded, marital_encoded, anxiety_encoded,
                panic_encoded, treatment_encoded
            ]])
            input_scaled = scaler.transform(input_data)
            _model_output = model.predict(input_scaled)[0]

            st.divider()
            st.markdown("### 🌷 Your wellness summary")
            display_name = name.strip() if name.strip() else "there"
            st.markdown(f"#### You’ve got this, {display_name}.")
            st.write(
                "Thank you for checking in. Your answers can help you notice "
                "what has been feeling difficult and choose one small, "
                "manageable step. You do not have to change everything at once."
            )

            suggestions = []
            if stress in ["More than a week", "Nearly every day"]:
                suggestions.append((
                    "Manage stress in smaller steps",
                    "Write down the tasks on your mind, choose just one priority, "
                    "and work on it for 20–25 minutes before taking a short break."
                ))
            elif stress == "A few days":
                suggestions.append((
                    "Make space for short breaks",
                    "Try a brief walk, gentle stretching, or a few slow breaths "
                    "between study sessions."
                ))

            if low_mood in ["More than a week", "Nearly every day"]:
                suggestions.append((
                    "Stay connected",
                    "Consider telling a trusted friend, family member, mentor, "
                    "or counsellor how you have been feeling. You deserve support."
                ))
            if focus in ["More than a week", "Nearly every day"]:
                suggestions.append((
                    "Make studying easier to start",
                    "Choose one small task, keep your phone away for a short "
                    "focus session, and allow yourself to pause when needed."
                ))
            if sleep in ["Often disturbed", "Very difficult"]:
                suggestions.append((
                    "Support your sleep routine",
                    "Aim for a consistent sleep and wake time. Reduce stimulating "
                    "screen use before bed and create a calm wind-down routine."
                ))
            if energy in ["More than a week", "Nearly every day"]:
                suggestions.append((
                    "Look after your basic needs",
                    "Try regular meals, water, gentle movement and rest. If tiredness "
                    "continues or affects daily life, consider speaking with a health professional."
                ))
            if support == "Not at the moment":
                suggestions.append((
                    "Find one safe person to reach out to",
                    "If possible, start with someone you trust, a college mentor, "
                    "student-support service or counsellor. You do not need perfect words."
                ))
            elif support == "Sometimes":
                suggestions.append((
                    "Keep your support network close",
                    "Think of one person you could contact when you need someone to listen."
                ))
            if enjoyable == "Hardly / not yet":
                suggestions.append((
                    "Make room for something enjoyable",
                    "Set aside 10–15 minutes for music, drawing, a hobby, a walk, "
                    "or another activity that feels pleasant to you."
                ))
            elif enjoyable == "Sometimes":
                suggestions.append((
                    "Protect a little personal time",
                    "Try scheduling one small enjoyable activity this week, "
                    "just as you would schedule a class or study session."
                ))

            if not suggestions:
                suggestions.append((
                    "Keep building healthy routines",
                    "Continue making time for rest, supportive people, movement "
                    "and activities you enjoy. Small habits can make a difference."
                ))

            st.markdown("### 🌱 Suggestions based on your answers")
            for heading, detail in suggestions:
                st.markdown(
                    f'<div class="soft-card"><h3>{heading}</h3><p>{detail}</p></div>',
                    unsafe_allow_html=True
                )

            st.markdown("### 💙 A gentle reminder")
            st.write(
                "Your feelings matter, and asking for support is a positive step. "
                "If these difficulties persist, become overwhelming, or affect "
                "your daily life, consider speaking with a qualified mental-health "
                "professional or your college counselling service."
            )
            st.caption(
                "Privacy note: This app displays suggestions from the answers "
                "you submit in this session. This is an educational wellness "
                "reflection, not a diagnosis, clinical screening, or substitute "
                "for professional care. If you are in immediate danger, contact "
                "local emergency services or a trusted person nearby."
            )

        except Exception as error:
            st.error(
                "The check-in could not be processed. Please verify that the "
                "original model files and their trained categories are present."
            )
            st.caption(f"Technical details: {error}")

# -------------------------------------------------
# INSIGHTS
# -------------------------------------------------
elif page == "Insights":
    st.markdown("""
    <div class="hero">
        <h1>Dataset Insights 📊</h1>
        <p>Explore the response distributions available in the project dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    required_columns = [
        "Do you have Depression?",
        "Do you have Anxiety?"
    ]

    if all(column in df.columns for column in required_columns):
        total_rows = len(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Dataset records", f"{total_rows}")
        c2.metric(
            "Depression responses",
            f"{df['Do you have Depression?'].notna().sum()}"
        )
        c3.metric(
            "Anxiety responses",
            f"{df['Do you have Anxiety?'].notna().sum()}"
        )

        left, right = st.columns(2)
        with left:
            st.markdown("### Depression status")
            depression_count = df["Do you have Depression?"].value_counts()
            st.bar_chart(depression_count)
        with right:
            st.markdown("### Anxiety status")
            anxiety_count = df["Do you have Anxiety?"].value_counts()
            st.bar_chart(anxiety_count)

        with st.expander("View dataset preview"):
            st.dataframe(df.head(10), use_container_width=True)
    else:
        st.error("The expected depression/anxiety columns were not found in the CSV.")

# -------------------------------------------------
# WELLNESS ASSISTANT
# -------------------------------------------------
elif page == "Wellness Assistant":
    st.markdown("""
    <div class="hero">
        <h1>Wellness Assistant 💬</h1>
        <p>Ask for simple general suggestions about stress, sleep or exam pressure.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "This is a keyword-based helper with predefined responses. "
        "It is not a medical or emergency support service."
    )

    user_question = st.text_input(
        "What would you like help with?",
        placeholder="For example: I feel stressed about exams..."
    )

    if st.button("Get suggestion"):
        if not user_question.strip():
            st.warning("Please type a question first.")
        else:
            question = user_question.lower()

            if "stress" in question or "anxious" in question or "anxiety" in question:
                answer = (
                    "Try taking a few slow breaths, divide your tasks into "
                    "small steps, and take short breaks. If these feelings "
                    "continue or feel difficult to manage, consider talking "
                    "with someone you trust or a qualified professional."
                )
            elif "sleep" in question or "tired" in question:
                answer = (
                    "A consistent sleep and wake-up time may help. Try reducing "
                    "screen use before bedtime and give yourself time to relax."
                )
            elif "exam" in question or "study" in question or "pressure" in question:
                answer = (
                    "Make a realistic study plan, focus on one topic at a time, "
                    "and include short breaks. You do not need to finish everything "
                    "at once."
                )
            else:
                answer = (
                    "Take a moment to pause and look after yourself. Writing down "
                    "what is bothering you or speaking with a trusted person may help."
                )

            st.success(answer)

    st.caption(
        "If you may be in immediate danger or need urgent help, contact local "
        "emergency services or a trusted person nearby."
    )

# -------------------------------------------------
# ABOUT
# -------------------------------------------------
elif page == "About":
    st.markdown("""
    <div class="hero">
        <h1>About MindCare ℹ️</h1>
        <p>Student Mental Health Agent — a machine-learning based academic project.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Project objective")
    st.write(
        "The project explores how student-related information and selected "
        "mental-health questionnaire responses can be processed by a trained "
        "machine-learning model to produce a depression-class prediction."
    )

    st.markdown("### Main components")
    st.markdown("""
    - Student information and assessment form
    - Existing trained machine-learning model
    - Data scaling and label encoding
    - Dataset visualizations for depression and anxiety responses
    - Keyword-based wellness suggestion assistant
    """)

    st.markdown("### Technologies used")
    st.markdown("**Python · Streamlit · Pandas · NumPy · Scikit-learn · Joblib**")

    st.warning(
        "This project is intended for educational awareness and demonstration. "
        "It does not diagnose depression, replace professional assessment, "
        "or provide medical advice."
    )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(
    '<div class="footer">MindCare · Student Mental Wellness Project</div>',
    unsafe_allow_html=True
)
