from pathlib import Path

import os



import joblib

import pandas as pd

import plotly.express as px

import streamlit as st

from dotenv import load_dotenv

from pymongo import MongoClient





# --------------------------------------------------

# Paths and configuration

# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env")



DATA_PATH = PROJECT_DIR / "data" / "cleaned_students.csv"

RESULTS_PATH = PROJECT_DIR / "results" / "model_comparison.csv"

MODEL_DIR = PROJECT_DIR / "results"



st.set_page_config(

    page_title="Student Performance Analytics",

    page_icon="🎓",

    layout="wide",

)





# --------------------------------------------------

# --------------------------------------------------
# Styling — dark navy analytics theme
# --------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #0B1120;
        --surface: #111B2E;
        --surface-raised: #162238;
        --border: #263650;
        --text: #E5EDF9;
        --muted: #94A3B8;
        --blue: #3B82F6;
        --teal: #2DD4BF;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0.92);
    }

    [data-testid="stToolbar"] { right: 1rem; }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }

    h1, h2, h3, h4, p, label, li,
    [data-testid="stMarkdownContainer"] {
        color: var(--text);
    }

    h1 {
        font-weight: 750;
        letter-spacing: -0.035em;
    }

    h2, h3, h4 { letter-spacing: -0.02em; }

    .subtitle {
        color: var(--muted);
        font-size: 1rem;
        margin-top: -0.35rem;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #142039, #10192B);
        border: 1px solid var(--border);
        padding: 18px 20px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }

    [data-testid="stMetricValue"] {
        color: #F1F5F9;
        font-weight: 750;
    }

    [data-testid="stMetricDelta"] { color: var(--teal); }

    [data-testid="stSidebar"] {
        background: #080D19;
        border-right: 1px solid #1D2A40;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: transparent;
        border-radius: 9px;
        padding: 0.35rem 0.5rem;
    }

    div[data-testid="stForm"],
    [data-testid="stExpander"],
    [data-testid="stTabs"] [data-baseweb="tab-panel"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 15px;
    }

    div[data-testid="stForm"] { padding: 22px; }

    div.stButton > button,
    div.stFormSubmitButton > button,
    [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #2563EB, #3B82F6);
        color: #FFFFFF;
        border: 1px solid #4B8CF8;
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.65rem;
        transition: all 150ms ease;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover,
    [data-testid="stDownloadButton"] > button:hover {
        background: #1D4ED8;
        color: #FFFFFF;
        border-color: #60A5FA;
        transform: translateY(-1px);
    }

    input, textarea,
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {
        background-color: #0E1728 !important;
        color: var(--text) !important;
        border-color: #34445F !important;
        border-radius: 9px !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        background: #111B2E !important;
        border-color: var(--border) !important;
    }

    [role="option"] { color: var(--text) !important; }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stTabs"] button[role="tab"] {
        color: var(--muted);
        font-weight: 600;
    }

    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--teal);
        border-bottom-color: var(--teal);
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid var(--border);
    }

    hr {
        border-color: #263650;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--blue), var(--teal));
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.25rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        div[data-testid="stForm"] { padding: 14px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load local data, results and models

# --------------------------------------------------

@st.cache_data

def load_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(

            f"Dataset not found: {DATA_PATH}. "

            "Run the preprocessing script first."

        )

    return pd.read_csv(DATA_PATH)





@st.cache_data

def load_results():

    if not RESULTS_PATH.exists():

        raise FileNotFoundError(

            f"Model comparison file not found: {RESULTS_PATH}. "

            "Run src/train_models.py first."

        )

    return pd.read_csv(RESULTS_PATH)





@st.cache_resource

def load_model(model_path):

    return joblib.load(model_path)





@st.cache_resource
def get_mongo_client():
    uri = os.getenv("MONGODB_URI")

    if not uri:
        try:
            uri = st.secrets["MONGODB_URI"]
        except Exception:
            uri = None

    if not uri:
        raise ValueError(
            "MONGODB_URI was not found. Configure it in your "
            ".env file locally or in Streamlit Cloud Secrets."
        )

    return MongoClient(uri, serverSelectionTimeoutMS=10000)





try:

    df = load_data()

    results = load_results()

except Exception as error:

    st.error(f"Could not load project files: {error}")

    st.info(

        "Check that data/cleaned_students.csv and "

        "results/model_comparison.csv exist in your project folder."

    )

    st.stop()





# --------------------------------------------------

# Sidebar navigation

# --------------------------------------------------

st.sidebar.title("Student Analytics")

st.sidebar.caption("Academic performance insights")



page = st.sidebar.radio(

    "Navigate",

    [

        "Overview",

        "Model Comparison",

        "Dataset Explorer",

        "Predict Student",

        "MongoDB Records",

    ],

)





# --------------------------------------------------

# Header and KPIs

# --------------------------------------------------

st.title("Student Performance Analytics")

st.markdown(

    '<p class="subtitle">Explore student outcomes, academic patterns, '

    'and machine-learning predictions.</p>',

    unsafe_allow_html=True,

)



total = len(df)

pass_count = int((df["result"] == 1).sum()) if "result" in df.columns else 0

fail_count = int((df["result"] == 0).sum()) if "result" in df.columns else 0

pass_rate = (pass_count / total * 100) if total else 0



c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Students", f"{total:,}")

c2.metric("Pass Rate", f"{pass_rate:.1f}%")

c3.metric("Passed", f"{pass_count:,}")

c4.metric("Failed", f"{fail_count:,}")



st.divider()





# ==================================================

# PAGE 1: OVERVIEW

# ==================================================

if page == "Overview":

    left, right = st.columns(2)



    with left:

        st.subheader("Pass vs Fail")

        result_counts = (

            df["result"]

            .map({0: "Fail", 1: "Pass"})

            .value_counts()

            .rename_axis("Result")

            .reset_index(name="Students")

        )



        fig = px.pie(

            result_counts,

            names="Result",

            values="Students",

            hole=0.62,

            color="Result",

            color_discrete_map={"Pass": "#0f766e", "Fail": "#e76f51"},

        )

        fig.update_traces(textposition="inside", textinfo="percent+label")

        fig.update_layout(

            paper_bgcolor="#111B2E",

            plot_bgcolor="#111B2E",

            font_color="#E5EDF9",

            margin=dict(t=25, b=20, l=10, r=10),

            legend_title_text="",

        )

        st.plotly_chart(fig, use_container_width=True)



    with right:

        st.subheader("Final Grade Distribution")

        if "G3" in df.columns:

            fig = px.histogram(

                df,

                x="G3",

                nbins=21,

                color_discrete_sequence=["#2563eb"],

                labels={"G3": "Final Grade", "count": "Students"},

            )

            fig.add_vline(

                x=10,

                line_dash="dash",

                line_color="#e76f51",

                annotation_text="Pass threshold (10)",

                annotation_position="top",

            )

            fig.update_layout(

                paper_bgcolor="#111B2E",

                plot_bgcolor="#111B2E",

                font_color="#E5EDF9",

                xaxis_title="Final Grade (0–20)",

                yaxis_title="Number of Students",

                showlegend=False,

            )

            st.plotly_chart(fig, use_container_width=True)

        else:

            st.info("The G3 column is not present in the dataset.")



    left, right = st.columns(2)



    with left:

        st.subheader("Study Time vs Final Grade")

        if {"studytime", "G3", "result"}.issubset(df.columns):

            fig = px.box(

                df,

                x="studytime",

                y="G3",

                color="result",

                color_discrete_map={0: "#e76f51", 1: "#0f766e"},

                labels={

                    "studytime": "Study Time Category (1–4)",

                    "G3": "Final Grade",

                    "result": "Result (0 = Fail, 1 = Pass)",

                },

            )

            fig.update_layout(

                paper_bgcolor="#111B2E",

                plot_bgcolor="#111B2E",

                font_color="#E5EDF9",

            )

            st.plotly_chart(fig, use_container_width=True)

        else:

            st.info("Required columns for this chart are missing.")



    with right:

        st.subheader("Absences vs Final Grade")

        if {"absences", "G3", "result"}.issubset(df.columns):

            fig = px.scatter(

                df,

                x="absences",

                y="G3",

                color="result",

                opacity=0.7,

                color_discrete_map={0: "#e76f51", 1: "#0f766e"},

                labels={

                    "absences": "Number of Absences",

                    "G3": "Final Grade",

                    "result": "Result (0 = Fail, 1 = Pass)",

                },

            )

            fig.update_layout(

                paper_bgcolor="#111B2E",

                plot_bgcolor="#111B2E",

                font_color="#E5EDF9",

            )

            st.plotly_chart(fig, use_container_width=True)

        else:

            st.info("Required columns for this chart are missing.")



    st.caption(

        "These charts show associations in the dataset; they do not establish causation."

    )





# ==================================================

# PAGE 2: MODEL COMPARISON

# ==================================================

elif page == "Model Comparison":

    st.subheader("Machine Learning Model Performance")



    metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score"]

    missing_metrics = [col for col in ["Model", *metric_columns] if col not in results.columns]



    if missing_metrics:

        st.error(

            "The model comparison file is missing these columns: "

            + ", ".join(missing_metrics)

        )

    else:

        display_results = results.copy()

        display_results[metric_columns] = (

            display_results[metric_columns] * 100

        ).round(1)



        st.dataframe(

            display_results,

            use_container_width=True,

            hide_index=True,

        )



        plot_df = results.melt(

            id_vars="Model",

            value_vars=metric_columns,

            var_name="Metric",

            value_name="Score",

        )



        fig = px.bar(

            plot_df,

            x="Metric",

            y="Score",

            color="Model",

            barmode="group",

            text_auto=".1%",

            color_discrete_sequence=["#2563eb", "#0f766e"],

        )

        fig.update_layout(

            paper_bgcolor="#111B2E",

            plot_bgcolor="#111B2E",

            font_color="#E5EDF9",

            yaxis=dict(tickformat=".0%", range=[0, 1]),

            yaxis_title="Score",

            xaxis_title="",

            legend_title_text="Model",

        )

        st.plotly_chart(fig, use_container_width=True)



        st.info(

            "The models were evaluated on one 80/20 train-test split. "

            "Random Forest had a higher overall F1-score in this run, but "

            "its recall for failing students was low. These results are preliminary."

        )



        st.markdown("#### Interpretation")

        st.write(

            "Accuracy is the proportion of correct predictions. Precision "

            "is the proportion of predicted positive outcomes that are correct. "

            "Recall is the proportion of actual positive outcomes identified. "

            "F1-score balances precision and recall for the positive (pass) class."

        )

        st.warning(

            "Fail-class recall must also be considered before using either "

            "model as an early-warning tool."

        )





# ==================================================

# PAGE 3: DATASET EXPLORER

# ==================================================

elif page == "Dataset Explorer":

    st.subheader("Explore Student Records")



    search_col, result_col = st.columns(2)



    with search_col:

        min_grade = st.slider(

            "Minimum final grade",

            min_value=0,

            max_value=20,

            value=0,

        )



    with result_col:

        outcome = st.selectbox("Filter by result", ["All", "Pass", "Fail"])



    filtered = df.copy()

    if "G3" in filtered.columns:

        filtered = filtered[filtered["G3"] >= min_grade]



    if outcome == "Pass":

        filtered = filtered[filtered["result"] == 1]

    elif outcome == "Fail":

        filtered = filtered[filtered["result"] == 0]



    st.write(f"Showing **{len(filtered)}** student records")

    st.dataframe(filtered, use_container_width=True, hide_index=True)



    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(

        "Download filtered records as CSV",

        data=csv_data,

        file_name="filtered_student_records.csv",

        mime="text/csv",

    )





# ==================================================

# PAGE 4: PREDICT STUDENT

# ==================================================

elif page == "Predict Student":

    st.subheader("Predict Student Outcome")

    st.write(

        "Enter student details to estimate the likelihood of passing. "

        "The model does not use first, second, or final grades as prediction inputs."

    )



    model_choice = st.selectbox(

        "Choose a model",

        ["Random Forest", "Logistic Regression"],

    )



    with st.form("student_prediction_form"):

        st.markdown("#### Academic Information")

        col1, col2, col3 = st.columns(3)



        with col1:

            age = st.number_input("Age", min_value=15, max_value=22, value=17)

            studytime = st.selectbox(

                "Weekly study time",

                [1, 2, 3, 4],

                format_func=lambda x: {

                    1: "Less than 2 hours",

                    2: "2–5 hours",

                    3: "5–10 hours",

                    4: "More than 10 hours",

                }[x],

            )

            failures = st.number_input(

                "Past class failures", min_value=0, max_value=4, value=0

            )

            absences = st.number_input(

                "School absences", min_value=0, max_value=93, value=0

            )



        with col2:

            schoolsup = st.selectbox("Extra educational support", ["no", "yes"])

            famsup = st.selectbox("Family educational support", ["yes", "no"])

            paid = st.selectbox("Extra paid classes", ["no", "yes"])

            internet = st.selectbox("Internet access at home", ["yes", "no"])



        with col3:

            school = st.selectbox("School", ["GP", "MS"])

            sex = st.selectbox("Sex", ["F", "M"])

            activities = st.selectbox("Extracurricular activities", ["yes", "no"])

            higher = st.selectbox("Wants higher education", ["yes", "no"])



        st.caption(

            "Other model inputs use the most common values from the dataset. "

            "This means the prediction is only partially personalized."

        )

        submitted = st.form_submit_button(

            "Predict Outcome",

            use_container_width=True,

        )



    if submitted:

        model_file = (

            "random_forest.joblib"

            if model_choice == "Random Forest"

            else "logistic_regression.joblib"

        )

        model_path = MODEL_DIR / model_file



        if not model_path.exists():

            st.error(

                f"Model file not found: {model_path}. Run src/train_models.py first."

            )

        else:

            try:

                model = load_model(str(model_path))



                feature_columns = [

                    col for col in df.columns

                    if col not in ["student_id", "G1", "G2", "G3", "result"]

                ]



                input_row = (

                    df[feature_columns]

                    .mode(dropna=False)

                    .iloc[[0]]

                    .copy()

                )



                supplied_values = {

                    "age": age,

                    "studytime": studytime,

                    "failures": failures,

                    "absences": absences,

                    "schoolsup": schoolsup,

                    "famsup": famsup,

                    "paid": paid,

                    "internet": internet,

                    "school": school,

                    "sex": sex,

                    "activities": activities,

                    "higher": higher,

                }



                for column, value in supplied_values.items():

                    if column in input_row.columns:

                        input_row.loc[:, column] = value



                prediction = int(model.predict(input_row)[0])

                probabilities = model.predict_proba(input_row)[0]

                classes = list(model.classes_)



                pass_probability = float(probabilities[classes.index(1)])

                fail_probability = float(probabilities[classes.index(0)])



                st.divider()

                st.markdown("### Prediction Result")

                result_col, probability_col = st.columns(2)



                with result_col:

                    if prediction == 1:

                        st.success("Predicted outcome: PASS")

                    else:

                        st.error("Predicted outcome: FAIL")



                with probability_col:

                    st.metric(

                        "Estimated pass probability",

                        f"{pass_probability:.1%}",

                    )



                st.progress(pass_probability)

                st.write(

                    f"Estimated fail probability: **{fail_probability:.1%}**"

                )

                st.warning(

                    "This estimate comes from a model trained on a small historical "

                    "dataset. Fields not shown in the form use typical dataset values. "

                    "Probabilities may not be well calibrated and are not guarantees."

                )



            except Exception as error:

                st.error(f"Prediction failed: {error}")





# ==================================================

# PAGE 5: MONGODB RECORDS + CRUD

# ==================================================

elif page == "MongoDB Records":

    
    # Password protection for MongoDB CRUD operations
    if "mongo_authenticated" not in st.session_state:
        st.session_state["mongo_authenticated"] = False

    if not st.session_state["mongo_authenticated"]:
        st.subheader("MongoDB Records — Restricted Access")
        st.write("Enter the password to access student database records.")

        entered_password = st.text_input(
            "CRUD access password",
            type="password",
            key="mongo_password_input",
        )

        if st.button("Unlock MongoDB Records"):
            expected_password = os.getenv("CRUD_PASSWORD")

            if not expected_password:
                try:
                    expected_password = st.secrets["CRUD_PASSWORD"]
                except Exception:
                    expected_password = None

            if not expected_password:
                st.error(
                    "CRUD_PASSWORD is not configured. "
                    "Set it in your local .env or Streamlit Cloud Secrets."
                )
            elif entered_password == expected_password:
                st.session_state["mongo_authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password. Try again.")

        st.stop()

    if st.button("Lock MongoDB Records"):
        st.session_state["mongo_authenticated"] = False
        st.rerun()

    st.subheader("MongoDB Atlas: Student Records")

    st.write(

        "View records and demonstrate Create, Read, Update and Delete (CRUD) "

        "operations directly against MongoDB Atlas."

    )



    try:

        client = get_mongo_client()

        client.admin.command("ping")



        db = client["student_performance_db"]

        collection = db["students"]



        st.success("Connected to MongoDB Atlas")



        total_records = collection.count_documents({})

        st.metric("Total database records", total_records)



        tab_read, tab_create, tab_update, tab_delete = st.tabs(

            ["Read records", "Create record", "Update record", "Delete record"]

        )



        # ------------------------------

        # READ

        # ------------------------------

        with tab_read:

            st.markdown("#### Read student records")

            limit = st.selectbox(

                "Number of records to display",

                [10, 25, 50, 100, 395],

                index=1,

                key="mongo_read_limit",

            )



            if st.button("Refresh records", key="mongo_refresh_records"):

                st.session_state["mongo_read_requested"] = True



            if (

                st.session_state.get("mongo_read_requested", False)

                or "mongo_read_requested" not in st.session_state

            ):

                records = list(

                    collection.find({}, {"_id": 0}).limit(int(limit))

                )



                if records:

                    mongo_df = pd.DataFrame(records)

                    st.dataframe(

                        mongo_df,

                        use_container_width=True,

                        hide_index=True,

                    )

                    st.download_button(

                        "Download displayed records",

                        data=mongo_df.to_csv(index=False).encode("utf-8"),

                        file_name="mongodb_student_records.csv",

                        mime="text/csv",

                        key="download_mongo_records",

                    )

                else:

                    st.info("No student records found.")



        # IDs are fetched from MongoDB so updates/deletes target an actual record.

        id_documents = list(

            collection.find(

                {"student_id": {"$exists": True}},

                {"_id": 0, "student_id": 1},

            )

        )

        student_ids = []

        seen_id_labels = set()

        for item in id_documents:

            student_id_value = item.get("student_id")

            label = f"{type(student_id_value).__name__}:{student_id_value}"

            if label not in seen_id_labels:

                student_ids.append(student_id_value)

                seen_id_labels.add(label)



        # ------------------------------

        # CREATE

        # ------------------------------

        with tab_create:

            st.markdown("#### Create a student record")

            st.caption(

                "To keep the document compatible with the dataset schema, "

                "fields not shown here are filled with typical values from the "

                "existing dataset. Use a new student ID."

            )



            with st.form("mongo_create_form"):

                new_student_id = st.text_input(

                    "New student ID",

                    value="NEW_STUDENT_001",

                    help="Use an ID that does not already exist.",

                )

                create_col1, create_col2, create_col3 = st.columns(3)



                with create_col1:

                    create_school = st.selectbox(

                        "School", ["GP", "MS"], key="create_school"

                    )

                    create_sex = st.selectbox(

                        "Sex", ["F", "M"], key="create_sex"

                    )

                    create_age = st.number_input(

                        "Age", min_value=15, max_value=22, value=17,

                        key="create_age",

                    )

                    create_studytime = st.selectbox(

                        "Weekly study time category",

                        [1, 2, 3, 4],

                        key="create_studytime",

                    )



                with create_col2:

                    create_failures = st.number_input(

                        "Past class failures", min_value=0, max_value=4,

                        value=0, key="create_failures",

                    )

                    create_absences = st.number_input(

                        "Absences", min_value=0, max_value=93,

                        value=0, key="create_absences",

                    )

                    create_schoolsup = st.selectbox(

                        "Extra educational support", ["no", "yes"],

                        key="create_schoolsup",

                    )

                    create_famsup = st.selectbox(

                        "Family educational support", ["yes", "no"],

                        key="create_famsup",

                    )



                with create_col3:

                    create_paid = st.selectbox(

                        "Extra paid classes", ["no", "yes"], key="create_paid"

                    )

                    create_internet = st.selectbox(

                        "Internet access at home", ["yes", "no"],

                        key="create_internet",

                    )

                    create_activities = st.selectbox(

                        "Extracurricular activities", ["yes", "no"],

                        key="create_activities",

                    )

                    create_higher = st.selectbox(

                        "Wants higher education", ["yes", "no"],

                        key="create_higher",

                    )



                create_submitted = st.form_submit_button(

                    "Create student record",

                    use_container_width=True,

                )



            if create_submitted:

                new_student_id = new_student_id.strip()



                if not new_student_id:

                    st.error("Student ID cannot be blank.")

                elif collection.find_one({"student_id": new_student_id}):

                    st.error(

                        "That student ID already exists. Choose a different ID."

                    )

                else:

                    try:

                        # Start with one typical row so the document has the

                        # same fields as the source dataset.

                        feature_source = df.drop(

                            columns=["result"], errors="ignore"

                        )

                        baseline = (

                            feature_source.mode(dropna=False)

                            .iloc[0]

                            .to_dict()

                        )



                        document = {}

                        for key, value in baseline.items():

                            if key == "_id":

                                continue

                            if pd.isna(value):

                                document[key] = None

                            elif hasattr(value, "item"):

                                document[key] = value.item()

                            else:

                                document[key] = value



                        document["student_id"] = new_student_id

                        document.update(

                            {

                                "school": create_school,

                                "sex": create_sex,

                                "age": int(create_age),

                                "studytime": int(create_studytime),

                                "failures": int(create_failures),

                                "absences": int(create_absences),

                                "schoolsup": create_schoolsup,

                                "famsup": create_famsup,

                                "paid": create_paid,

                                "internet": create_internet,

                                "activities": create_activities,

                                "higher": create_higher,

                            }

                        )



                        # result is derived during preprocessing and is not

                        # stored in the original MongoDB source documents.

                        document.pop("result", None)

                        insert_result = collection.insert_one(document)

                        st.success(

                            f"Record created successfully. MongoDB ID: "

                            f"{insert_result.inserted_id}"

                        )

                        st.rerun()

                    except Exception as error:

                        st.error(f"Could not create record: {error}")



        # ------------------------------

        # UPDATE

        # ------------------------------

        with tab_update:

            st.markdown("#### Update an existing student record")



            if not student_ids:

                st.info(

                    "No records with a student_id were found to update."

                )

            else:

                with st.form("mongo_update_form"):

                    selected_update_id = st.selectbox(

                        "Choose the student ID to update",

                        student_ids,

                        format_func=lambda value: str(value),

                        key="selected_update_id",

                    )

                    update_col1, update_col2 = st.columns(2)



                    with update_col1:

                        updated_age = st.number_input(

                            "New age", min_value=15, max_value=22, value=17,

                            key="updated_age",

                        )

                        updated_studytime = st.selectbox(

                            "New study time category",

                            [1, 2, 3, 4],

                            key="updated_studytime",

                        )



                    with update_col2:

                        updated_failures = st.number_input(

                            "New past class failures", min_value=0, max_value=4,

                            value=0, key="updated_failures",

                        )

                        updated_absences = st.number_input(

                            "New absences", min_value=0, max_value=93,

                            value=0, key="updated_absences",

                        )



                    confirm_update = st.checkbox(

                        "I confirm I want to modify this database record.",

                        key="confirm_mongo_update",

                    )

                    update_submitted = st.form_submit_button(

                        "Update selected record",

                        use_container_width=True,

                    )



                if update_submitted:

                    if not confirm_update:

                        st.warning("Tick the confirmation box before updating.")

                    else:

                        update_result = collection.update_one(

                            {"student_id": selected_update_id},

                            {

                                "$set": {

                                    "age": int(updated_age),

                                    "studytime": int(updated_studytime),

                                    "failures": int(updated_failures),

                                    "absences": int(updated_absences),

                                }

                            },

                        )

                        if update_result.matched_count:

                            st.success(

                                f"Updated student {selected_update_id}. "

                                f"Fields changed: {update_result.modified_count}."

                            )

                            st.rerun()

                        else:

                            st.error(

                                "No record matched that student ID. Refresh the page."

                            )



        # ------------------------------

        # DELETE

        # ------------------------------

        with tab_delete:

            st.markdown("#### Delete a student record")

            st.warning(

                "Deleting a record permanently removes it from MongoDB Atlas. "

                "Use a test record if you only need to demonstrate deletion."

            )



            if not student_ids:

                st.info(

                    "No records with a student_id were found to delete."

                )

            else:

                with st.form("mongo_delete_form"):

                    selected_delete_id = st.selectbox(

                        "Choose the student ID to delete",

                        student_ids,

                        format_func=lambda value: str(value),

                        key="selected_delete_id",

                    )

                    confirm_delete = st.checkbox(

                        "I understand this permanently deletes the selected record.",

                        key="confirm_mongo_delete",

                    )

                    delete_submitted = st.form_submit_button(

                        "Delete selected record",

                        use_container_width=True,

                    )



                if delete_submitted:

                    if not confirm_delete:

                        st.warning("Tick the confirmation box before deleting.")

                    else:

                        delete_result = collection.delete_one(

                            {"student_id": selected_delete_id}

                        )

                        if delete_result.deleted_count == 1:

                            st.success(

                                f"Student record {selected_delete_id} was deleted."

                            )

                            st.rerun()

                        else:

                            st.error(

                                "No record was deleted. It may already have been removed."

                            )



    except Exception as error:

        st.error(

            "Could not connect to MongoDB. Check your .env connection string, "

            "Atlas network access, and database credentials."

        )

        st.caption(str(error))





# --------------------------------------------------

# Footer

# --------------------------------------------------

st.divider()

st.caption(

    "Student Performance Analytics | Academic project prototype | "

    "Predictions are estimates, not guarantees."

)