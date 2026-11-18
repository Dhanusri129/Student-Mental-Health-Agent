import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

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

# Individual wellness data is kept only for the current Streamlit browser session.
if "wellness_history" not in st.session_state:
    st.session_state.wellness_history = []

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
    /* AGE AND CGPA: dark field with white typed values for clear contrast */
    [data-testid="stNumberInput"] div[data-baseweb="input"],
    [data-testid="stNumberInput"] div[data-baseweb="input"] input {
        background-color: #30265F !important;
    }

    [data-testid="stNumberInput"] input,
    [data-testid="stNumberInput"] input:focus {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        caret-color: #FFFFFF !important;
        background-color: #30265F !important;
        border-color: #7561C9 !important;
        opacity: 1 !important;
    }

    [data-testid="stNumberInput"] button {
        color: #FFFFFF !important;
        background-color: #7561C9 !important;
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
        st.markdown("### 🧠 Basic questionnaire")
        st.caption(
            "These are the basic questionnaire fields,these will help to detect your current health status properly used by the trained "
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

            # A fuller, plain-language reflection. This is not a clinical score.
            st.markdown("### 🪷 Your personal reflection")
            reflection_items = [
                ("Stress", stress), ("Mood", low_mood),
                ("Concentration", focus), ("Sleep", sleep),
                ("Energy", energy), ("Social support", support),
                ("Enjoyable activities", enjoyable),
            ]
            reflection_df = pd.DataFrame(reflection_items, columns=["Area", "Your response"])
            st.dataframe(reflection_df, use_container_width=True, hide_index=True)

            attention = []
            strengths = []
            if stress == "Never":
                strengths.append("You reported that stress has not been frequent recently.")
            elif stress in ["More than a week", "Nearly every day"]:
                attention.append("Stress has been showing up often. Try breaking responsibilities into smaller steps and making room for recovery time.")
            if low_mood in ["More than a week", "Nearly every day"]:
                attention.append("You reported feeling low or emotionally drained often. Consider sharing this with someone you trust or a qualified counsellor.")
            elif low_mood == "Never":
                strengths.append("You reported that low mood has not been frequent during this check-in period.")
            if focus in ["More than a week", "Nearly every day"]:
                attention.append("Concentration has felt difficult often. Try one short task at a time, reduce distractions, and take planned breaks.")
            if sleep in ["Often disturbed", "Very difficult"]:
                attention.append("Sleep has been difficult. A steady bedtime routine and reduced screen use before bed may help.")
            elif sleep == "Very restful":
                strengths.append("You reported that your sleep has been restful. Try to protect the routine that is working for you.")
            if energy in ["More than a week", "Nearly every day"]:
                attention.append("Low energy has been frequent. Pay attention to rest, regular meals and hydration; seek professional advice if it continues.")
            if support == "Yes, definitely":
                strengths.append("You have someone you feel comfortable talking to. Staying connected is a valuable support habit.")
            elif support == "Not at the moment":
                attention.append("You may not currently feel you have someone to talk to. A trusted friend, family member, faculty mentor or college counsellor could be a starting point.")
            if enjoyable == "Yes, regularly":
                strengths.append("You are making time for enjoyable activities. Keep protecting some personal time in your routine.")
            elif enjoyable == "Hardly / not yet":
                attention.append("You have had little time for enjoyable activities. Try setting aside 10–15 minutes for something you like.")

            left, right = st.columns(2)
            with left:
                st.markdown("#### 🌟 Things you can build on")
                if strengths:
                    for item in strengths:
                        st.markdown(f"- {item}")
                else:
                    st.write("Your check-in does not highlight a clear strength in these selected areas yet. That is okay; small habits can be built gradually.")
            with right:
                st.markdown("#### 💜 Areas that may need care")
                if attention:
                    for item in attention:
                        st.markdown(f"- {item}")
                else:
                    st.write("Your selected answers do not point to a frequent difficulty in these areas right now. Keep checking in with yourself.")

            st.markdown("#### 🗓️ A simple 3-step plan for this week")
            action_plan = [
                "Choose one manageable study or personal task each morning.",
                "Take at least one short break away from your screen during study time.",
                "Connect with someone supportive or spend a little time on an activity you enjoy.",
            ]
            for step_no, step in enumerate(action_plan, start=1):
                st.markdown(f"**Step {step_no}.** {step}")

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

            # Save this student's check-in for the My Insights page.
            checkin = {
                "date": datetime.now().strftime("%d %b %Y, %I:%M %p"),
                "name": name.strip() if name.strip() else "Student",
                "stress": stress,
                "mood": low_mood,
                "focus": focus,
                "sleep": sleep,
                "energy": energy,
                "support": support,
                "enjoyable": enjoyable,
                "suggestions": suggestions.copy(),
            }
            st.session_state.wellness_history.append(checkin)
            st.session_state.latest_wellness = checkin

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
        <h1>Wellness Insights 📊</h1>
        <p>View your personal check-in history and explore overall patterns in the original student survey dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    personal_tab, overall_tab = st.tabs(["👤 Individual Insights", "🌍 Overall Dataset Insights"])

    with personal_tab:
        st.markdown("### Your personal wellness overview")
        history = st.session_state.get("wellness_history", [])
        if not history:
            st.markdown("""
            <div class="soft-card">
                <h3>🌱 Your personal insights will appear here</h3>
                <p>Complete the Assessment to see your responses, tailored suggestions and check-in history here.</p>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Your personal answers are kept only in this active app session. They are not added to the shared dataset charts.")
        else:
            latest = history[-1]
            st.caption(f"Latest check-in: {latest['date']} · Student: {latest['name']}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Stress", latest["stress"])
            m2.metric("Mood", latest["mood"])
            m3.metric("Sleep", latest["sleep"])
            m4.metric("Energy", latest["energy"])

            st.markdown("#### Your latest reflection")
            reflection = pd.DataFrame({
                "Area": ["Stress", "Low mood", "Concentration", "Sleep", "Energy", "Social support", "Enjoyable activities"],
                "Your response": [latest["stress"], latest["mood"], latest["focus"], latest["sleep"], latest["energy"], latest["support"], latest["enjoyable"]],
            })
            st.dataframe(reflection, use_container_width=True, hide_index=True)

            st.markdown("#### Suggestions selected for you")
            for heading, detail in latest["suggestions"]:
                st.markdown(f'<div class="soft-card"><h3>{heading}</h3><p>{detail}</p></div>', unsafe_allow_html=True)

            st.markdown("#### Your check-in history")
            history_rows = [{
                "Check-in": f"Check-in {i + 1}", "Date": item["date"],
                "Stress": item["stress"], "Mood": item["mood"],
                "Sleep": item["sleep"], "Energy": item["energy"],
            } for i, item in enumerate(history)]
            st.dataframe(pd.DataFrame(history_rows[::-1]), use_container_width=True, hide_index=True)
            if len(history) >= 2:
                scale = ["Never", "A few days", "More than a week", "Nearly every day"]
                trend = pd.DataFrame({
                    "Check-in": [f"Check-in {i + 1}" for i in range(len(history))],
                    "Stress frequency (response order)": [scale.index(item["stress"]) for item in history],
                    "Low-mood frequency (response order)": [scale.index(item["mood"]) for item in history],
                }).set_index("Check-in")
                st.markdown("#### Changes across your check-ins")
                st.line_chart(trend, use_container_width=True)
                st.caption("The chart displays the order of your selected response options, not a medical or clinical score.")
            st.info("Your personal insights are visible only in this active session. They may be cleared when the session ends or the app restarts.")

    with overall_tab:
        st.markdown("### Overall student survey patterns")
        st.caption("These charts use the original CSV dataset included with the project. They do not include the private check-ins entered by app users.")
        depression_col = "Do you have Depression?"
        anxiety_col = "Do you have Anxiety?"
        panic_col = "Do you have Panic attack?"

        if df.empty:
            st.warning("The dataset is empty, so overall insights cannot be displayed.")
        elif depression_col not in df.columns or anxiety_col not in df.columns:
            st.error("The expected depression and anxiety columns were not found in the CSV.")
            st.caption("Available columns: " + ", ".join(map(str, df.columns)))
        else:
            total_students = len(df)
            def yes_count(column):
                if column not in df.columns:
                    return None
                values = df[column].dropna().astype(str).str.strip().str.lower()
                return int(values.isin(["yes", "y", "true", "1"]).sum())
            def pct(count):
                return "—" if count is None or total_students == 0 else f"{count / total_students * 100:.1f}%"

            dep_yes, anx_yes, panic_yes = yes_count(depression_col), yes_count(anxiety_col), yes_count(panic_col)
            st.markdown("#### Dataset at a glance")
            a, b, c, d = st.columns(4)
            a.metric("Survey records", f"{total_students:,}")
            b.metric("Depression – Yes", pct(dep_yes))
            c.metric("Anxiety – Yes", pct(anx_yes))
            d.metric("Panic attack – Yes", pct(panic_yes))
            st.info("These are self-reported survey responses in the source dataset. They are not diagnoses and should not be interpreted as the prevalence of a clinical condition.")

            st.markdown("#### Response distributions")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Depression responses**")
                dep_dist = df[depression_col].fillna("Not answered").astype(str).value_counts().rename_axis("Response").to_frame("Students")
                st.bar_chart(dep_dist, use_container_width=True)
            with c2:
                st.markdown("**Anxiety responses**")
                anx_dist = df[anxiety_col].fillna("Not answered").astype(str).value_counts().rename_axis("Response").to_frame("Students")
                st.bar_chart(anx_dist, use_container_width=True)
            if panic_col in df.columns:
                st.markdown("**Panic attack responses**")
                panic_dist = df[panic_col].fillna("Not answered").astype(str).value_counts().rename_axis("Response").to_frame("Students")
                st.bar_chart(panic_dist, use_container_width=True)

            st.markdown("#### Student profile distributions")
            profile_columns = {
                "Gender": "Choose your gender", "Course": "What is your course?",
                "Year of study": "Your current year of Study", "Age": "Age",
            }
            available = [(label, col) for label, col in profile_columns.items() if col in df.columns]
            if available:
                profile_cols = st.columns(2)
                for idx, (label, col) in enumerate(available):
                    with profile_cols[idx % 2]:
                        st.markdown(f"**{label}**")
                        counts = df[col].fillna("Not answered").astype(str).value_counts().head(10).rename_axis(label).to_frame("Students")
                        st.bar_chart(counts, use_container_width=True)
            st.caption("Only aggregate counts and distributions are shown here. Individual source-dataset rows are not displayed or downloadable from this page.")

# -------------------------------------------------
# WELLNESS ASSISTANT
# -------------------------------------------------
elif page == "Wellness Assistant":
    st.markdown("""
    <div class="hero">
        <h1>Wellness Assistant 💜</h1>
        <p>A gentle space to check in with yourself, reset, and find small steps that may help.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
      .wellness-card {background:#ffffff; border:1px solid #e7e2fa; border-radius:18px;
        padding:20px 22px; margin:8px 0 16px 0; box-shadow:0 5px 18px rgba(65,52,120,.06);}
      .wellness-card h3 {color:#34266d !important; margin:0 0 8px 0;}
      .wellness-card p {color:#514b70 !important; margin:0;}
      .wellness-small {color:#625b7e !important; font-size:.92rem;}
      .wellness-tag {display:inline-block; background:#f0ecff; color:#44358b !important;
        padding:5px 10px; border-radius:20px; font-size:.82rem; font-weight:700;}
    </style>
    """, unsafe_allow_html=True)

    st.info("This page offers general wellbeing ideas, not a diagnosis or treatment. Choose only activities that feel comfortable for you.")

    mood_tab, reset_tab, balance_tab, support_tab = st.tabs([
        "🌤️ Mood check-in", "🌿 Relaxation corner", "📚 Study-life balance", "💛 Support corner"
    ])

    with mood_tab:
        st.markdown("### How are you feeling today?")
        st.write("There is no right or wrong answer. Pick the option that feels closest to your day.")
        mood = st.selectbox(
            "Choose your current mood",
            ["Select a mood", "Happy / positive", "Motivated", "Stressed / overwhelmed",
             "Worried / anxious", "Low / sad", "Tired / drained", "Not sure"],
            key="wellness_mood"
        )
        if mood != "Select a mood":
            mood_guidance = {
                "Happy / positive": ("Keep a little of this feeling with you", "Notice what helped today. You could write down one good moment or share it with someone you care about."),
                "Motivated": ("Turn motivation into one manageable step", "Choose one task that matters today, decide when to do it, and remember to take breaks rather than filling every minute."),
                "Stressed / overwhelmed": ("Make the next step smaller", "Pause for a few slow breaths. Write down what is on your mind, choose just one priority, and ask for help if the load feels too much."),
                "Worried / anxious": ("Bring your attention back to the present", "Try the 5-4-3-2-1 grounding activity in the Relaxation Corner. If worry keeps affecting daily life, consider speaking with a counsellor or qualified professional."),
                "Low / sad": ("You do not have to handle everything alone", "Try one gentle activity—getting some fresh air, having a meal, or messaging someone you trust. It is okay to ask for support."),
                "Tired / drained": ("Give yourself permission to recharge", "Check whether you need water, food, movement, rest, or a break from screens. If tiredness continues, consider discussing it with a health professional."),
                "Not sure": ("It is okay not to have a clear label", "Take a moment to notice what your body needs right now. You can write freely in the reflection box in the Support Corner.")
            }
            heading, suggestion = mood_guidance[mood]
            st.markdown(f'<div class="wellness-card"><span class="wellness-tag">YOUR CHECK-IN</span><h3>{heading}</h3><p>{suggestion}</p></div>', unsafe_allow_html=True)
            st.caption("Your mood selection is used only to show this guidance in the current page; it is not a clinical assessment.")

    with reset_tab:
        st.markdown("### Take a short pause")
        st.write("Choose a simple exercise. Stop if it feels uncomfortable, and breathe normally whenever you need to.")
        exercise = st.radio("Choose an activity", ["Calm breathing", "5-4-3-2-1 grounding", "Quick body reset"], horizontal=True)
        if exercise == "Calm breathing":
            st.markdown('<div class="wellness-card"><h3>🌬️ Gentle breathing</h3><p>1. Sit comfortably and relax your shoulders.<br>2. Breathe in gently for 4 counts.<br>3. Breathe out slowly for 4 to 6 counts.<br>4. Repeat for about one minute, without forcing your breath.</p></div>', unsafe_allow_html=True)
            st.caption("If focusing on breathing makes you uncomfortable, try the grounding exercise instead.")
        elif exercise == "5-4-3-2-1 grounding":
            st.markdown('<div class="wellness-card"><h3>🌱 Notice what is around you</h3><p>Look around and gently name:<br><b>5</b> things you can see · <b>4</b> things you can feel<br><b>3</b> things you can hear · <b>2</b> things you can smell<br><b>1</b> thing you can taste or one slow breath you notice.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="wellness-card"><h3>🙆 A quick body reset</h3><p>Unclench your jaw, lower your shoulders, stretch your hands, stand up if comfortable, and look away from your screen for a little while. Move gently—never push into pain.</p></div>', unsafe_allow_html=True)

    with balance_tab:
        st.markdown("### Make study feel more manageable")
        st.markdown('<div class="wellness-card"><h3>⏳ A simple focus cycle</h3><p>Pick one small task · Focus for 25 minutes · Take a 5-minute break · After four cycles, take a longer break. Adjust the timings to suit your needs.</p></div>', unsafe_allow_html=True)
        with st.expander("I have too many assignments"):
            st.write("List everything you need to do, mark what is due soonest, and start with a 10-minute action. If deadlines conflict, contact your lecturer early and explain clearly.")
        with st.expander("I feel pressure before exams"):
            st.write("Break revision into topics, practise a few questions, and protect time for meals and sleep. A realistic plan is more useful than an all-night study session.")
        with st.expander("I keep getting distracted by my phone"):
            st.write("Place your phone out of reach for one focus session, silence non-essential notifications, and decide in advance when you will check messages.")
        with st.expander("I am struggling to balance everything"):
            st.write("Choose your top two priorities for today. Leave some space for rest, family, friends, and activities you enjoy. You do not need to do everything at once.")

    with support_tab:
        st.markdown("### A little encouragement")
        st.markdown('<div class="wellness-card"><h3>💛 A reminder for today</h3><p>Progress does not have to be perfect or fast. Small steps count, and asking for support is a strength.</p></div>', unsafe_allow_html=True)
        st.markdown("#### Reflection journal")
        prompt = st.selectbox("Choose a reflection prompt", [
            "What is one thing taking up my energy today?",
            "What is one small thing I can do for myself today?",
            "Who is someone I feel comfortable talking to?",
            "What is one thing I handled well recently?"
        ])
        reflection = st.text_area("Write a few thoughts (optional)", placeholder="This space is for your own reflection...", height=110, key="wellness_reflection")
        st.caption("Your writing is not sent anywhere by this page. It is not saved as a permanent journal.")
        if st.button("Show my reflection prompt", key="show_reflection_prompt"):
            st.success(prompt)
            if reflection.strip():
                st.info("Thank you for taking a moment to reflect. Consider one kind, realistic next step for yourself.")

        st.markdown("#### Today's small wellness challenge")
        challenges = [
            "Take a 10-minute walk or stretch gently.",
            "Drink a glass of water and take a screen break.",
            "Send a kind message to someone you trust.",
            "Spend 10 minutes on a hobby you enjoy.",
            "Prepare a realistic plan for just tomorrow.",
            "Take a few minutes to tidy or refresh your study space.",
            "Try to keep a regular bedtime tonight."
        ]
        today_challenge = challenges[datetime.now().timetuple().tm_yday % len(challenges)]
        st.markdown(f'<div class="wellness-card"><span class="wellness-tag">ONE SMALL STEP</span><h3>{today_challenge}</h3><p>Choose it only if it works for you. Small, realistic actions are enough.</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### When extra support may help")
    st.write("If difficult feelings persist, become intense, or interfere with sleep, study, relationships, or daily activities, consider reaching out to a trusted person, your college counsellor, or a qualified mental-health professional. If you are in immediate danger or may hurt yourself, contact local emergency services or go to the nearest emergency department, and tell a trusted person nearby now.")
    st.caption("MindCare is an educational student-wellness project. It does not provide medical advice, diagnosis, counselling, or emergency response.")

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
