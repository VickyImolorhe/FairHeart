import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import joblib
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go

st.set_page_config(page_title="Fair Heart", layout="wide")

# ---------- lotus logo + styling ----------
_lotus_raw = """
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="0 0 100 100">
  <path d="M50 85 C50 60 50 40 50 30 C55 45 60 60 50 85 Z" fill="#e79bb0"/>
  <path d="M50 85 C50 60 50 40 50 30 C45 45 40 60 50 85 Z" fill="#d98aa1"/>
  <path d="M50 85 C35 65 25 50 20 42 C35 48 48 62 50 85 Z" fill="#7fc4c4"/>
  <path d="M50 85 C65 65 75 50 80 42 C65 48 52 62 50 85 Z" fill="#7fc4c4"/>
  <path d="M50 85 C25 72 12 62 6 56 C24 58 42 68 50 85 Z" fill="#2b8a8a"/>
  <path d="M50 85 C75 72 88 62 94 56 C76 58 58 68 50 85 Z" fill="#2b8a8a"/>
</svg>
"""
_b64 = base64.b64encode(_lotus_raw.encode()).decode()

st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/svg+xml;base64,{_b64}");
    background-repeat: no-repeat;
    background-position: center 40%;
    background-size: 360px;
    background-attachment: fixed;
}}
.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(242,248,248,0.80);
    z-index: 0;
    pointer-events: none;
}}
.block-container {{ position: relative; z-index: 1; }}
[data-testid="stSidebar"] {{ z-index: 2; }}
.stButton button, [data-testid="stForm"] button {{
    min-width: 150px;
    background-color: #2b8a8a !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.4rem !important;
}}
.stButton button:hover, [data-testid="stForm"] button:hover {{
    background-color: #216d6d !important;
}}
h1, h2, h3 {{ color: #226f6f; }}
</style>
""", unsafe_allow_html=True)


def welcome_header():
    st.markdown(f"""
    <div style="text-align:center; margin-bottom:1rem;">
      <img src="data:image/svg+xml;base64,{_b64}" width="90"/>
      <h1 style="color:#2b8a8a; margin-bottom:0;">Welcome to Fair Heart</h1>
      <p style="color:#5a7a7a; font-size:1.1rem; margin-top:0.2rem;">
        Fairness-aware heart disease risk prediction</p>
    </div>
    """, unsafe_allow_html=True)


# ---------- users / login ----------
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

if st.session_state.get("authentication_status") is None:
    welcome_header()

authenticator.login()

if st.session_state.get("authentication_status") is False:
    st.error("Username or password is incorrect.")
    st.stop()
elif st.session_state.get("authentication_status") is None:
    st.warning("Please enter your username and password to continue.")
    st.stop()

# ---------- logged in ----------
name = st.session_state.get("name")
username = st.session_state.get("username")
role = config["credentials"]["usernames"][username]["roles"]

st.sidebar.success(f"Logged in as: {name}")
st.sidebar.caption(f"Role: {role}")
authenticator.logout("Log out", "sidebar")

# ---------- load models ----------
@st.cache_resource
def load_everything():
    features = joblib.load("feature_names.joblib")
    results = pd.read_json("results.json")
    models = {
        "Baseline (no fix)":           joblib.load("model_baseline.joblib"),
        "Reweighting (pre)":           joblib.load("model_reweight.joblib"),
        "Exponentiated Gradient (in)": joblib.load("model_expgrad.joblib"),
        "Threshold Optimizer (post)":  joblib.load("model_threshold.joblib"),
    }
    return features, results, models
features, results, MODELS = load_everything()
PROB_MODELS = ["Baseline (no fix)", "Reweighting (pre)"]

features, results, MODELS = load_everything()

# ---------- header ----------
st.title("Fair Heart")
st.markdown("##### Detecting and reducing sex bias in heart disease risk prediction")
st.caption("An educational demonstration — not a clinical diagnostic tool.")

# ---------- sidebar inputs ----------
st.sidebar.header("Patient details")
age      = st.sidebar.slider("Age", 20, 90, 54)
sex_lbl  = st.sidebar.radio("Sex", ["Female", "Male"], horizontal=True)
cp       = st.sidebar.slider("Chest pain type (1-4)", 1, 4, 4)
trestbps = st.sidebar.slider("Resting blood pressure", 90, 200, 130)
chol     = st.sidebar.slider("Cholesterol", 100, 600, 240)
fbs      = st.sidebar.radio("Fasting blood sugar > 120 mg/dl?", [0, 1], horizontal=True)
restecg  = st.sidebar.slider("Resting ECG result (0-2)", 0, 2, 1)
thalach  = st.sidebar.slider("Maximum heart rate", 70, 210, 150)
exang    = st.sidebar.radio("Exercise-induced angina?", [0, 1], horizontal=True)
oldpeak  = st.sidebar.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)


def get_prediction(model_name, sex_value):
    data = {"age": age, "sex": sex_value, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak}
    row = pd.DataFrame([[data[f] for f in features]], columns=features)
    obj = MODELS[model_name]
    if model_name in PROB_MODELS:
        return ("prob", obj.predict_proba(row)[0][1] * 100)
    scaler = obj["scaler"]; model = obj["model"]
    row_scaled = scaler.transform(row)
    if model_name == "Threshold Optimizer (post)":
        pred = model.predict(row_scaled, sensitive_features=np.array([sex_value]))
    else:
        pred = model.predict(row_scaled)
    return ("yesno", int(pred[0]))


def show_value(kind, value):
    if kind == "prob":
        return f"{value:.1f}%"
    return "Yes — disease predicted" if value == 1 else "No — no disease"


# ---------- tabs (role-based) ----------
tab_names = ["What-if tool", "Fairness audit", "About", "My account"]
if role == "admin":
    tab_names.append("User management")
tabs = st.tabs(tab_names)

# --- What-if tool ---
with tabs[0]:
    st.subheader("Individual risk — the what-if tool")
    chosen = st.selectbox("Choose the model:", list(MODELS.keys()))
    compare = st.toggle("Compare Female vs Male for this exact patient")
    if not compare:
        sex_value = 0.0 if sex_lbl == "Female" else 1.0
        kind, val = get_prediction(chosen, sex_value)
        st.metric(f"Result (recorded as {sex_lbl})", show_value(kind, val))
    else:
        f_kind, f_val = get_prediction(chosen, 0.0)
        m_kind, m_val = get_prediction(chosen, 1.0)
        c1, c2 = st.columns(2)
        c1.metric("Same patient as Female", show_value(f_kind, f_val))
        c2.metric("Same patient as Male",   show_value(m_kind, m_val))
        if f_kind == "prob":
            gap = abs(m_val - f_val)
            if gap >= 10:
                st.error(f"Changing only the sex shifts the risk by {gap:.1f} points — bias.")
            elif gap >= 3:
                st.warning(f"Sex-only difference: {gap:.1f} points.")
            else:
                st.success(f"Sex-only difference: {gap:.1f} points — fairer.")
    st.caption("Tip: keep the health sliders fixed and change only Sex to see the bias clearly.")

# --- Fairness audit ---
with tabs[1]:
    st.subheader("Model comparison — the fairness audit")
    st.write("Each model's disease-detection rate for women and men, and the gap between them. "
             "A smaller gap means the model is fairer.")

    order = ["Baseline (no fix)", "Reweighting (pre)",
             "Exponentiated Gradient (in)", "Threshold Optimizer (post)"]
    results["order"] = results["model"].apply(lambda m: order.index(m) if m in order else 99)
    rs = results.sort_values("order").drop(columns="order").reset_index(drop=True)

    fig = go.Figure()
    fig.add_bar(name="Female recall", x=rs["model"], y=rs["female_recall"],
                marker_color="#FF6347", text=rs["female_recall"], textposition="outside")
    fig.add_bar(name="Male recall", x=rs["model"], y=rs["male_recall"],
                marker_color="#aed6f1", text=rs["male_recall"], textposition="outside")
    fig.update_layout(barmode="group", yaxis_title="Disease-detection rate (%)",
                      yaxis_range=[0, 100],
                      legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                      height=440, margin=dict(t=30, b=80),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.4)")
    st.plotly_chart(fig, use_container_width=True)

    st.write("**The fairness gap (male minus female), across the four models:**")
    fig2 = go.Figure()
    fig2.add_scatter(x=rs["model"], y=rs["gap"], mode="lines+markers+text",
                     text=[f"{g:.1f}" for g in rs["gap"]], textposition="top center",
                     line=dict(color="#5499c7", width=3), marker=dict(size=10, color="#5499c7"))
    fig2.update_layout(yaxis_title="Gap (percentage points)",
                       yaxis_range=[0, max(rs["gap"]) + 5], height=320, margin=dict(t=20, b=40),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.4)")
    st.plotly_chart(fig2, use_container_width=True)

    st.write("**Full results table:**")
    st.dataframe(rs, hide_index=True, use_container_width=True)
    st.caption("Lower gap = fairer. Results rest on relatively few women, so figures carry uncertainty.")

# --- About ---
with tabs[2]:
    st.subheader("About this project")
    st.write("Fair Heart investigates sex bias in machine-learning models that predict heart "
             "disease, and compares methods for reducing it. It pools four public UCI heart "
             "datasets, trains a baseline and three fairness-aware models, and shows the effect "
             "on individual patients.")
    st.caption("Educational and research use only. Not for clinical decision-making.")
# --- My account (change own password) ---
with tabs[3]:
    st.subheader("My account")
    st.write(f"Logged in as **{name}** (role: {role}).")
    st.write("Change your password below.")
    try:
        if authenticator.reset_password(username, location="main"):
            with open("config.yaml", "w") as file:
                yaml.dump(config, file, default_flow_style=False)
            st.success("Password changed successfully. Use it next time you log in.")
    except Exception as e:
        st.error(f"Could not change password: {e}")

# --- User management (admin only) ---
if role == "admin":
    with tabs[4]:
        st.subheader("User management (admin only)")

        # show current users
        st.write("**Current users:**")
        users = config["credentials"]["usernames"]
        user_table = pd.DataFrame([
            {"Username": u, "Name": info["name"], "Role": info["roles"]}
            for u, info in users.items()
        ])
        st.dataframe(user_table, hide_index=True, use_container_width=True)

        st.divider()

        # --- add a new user ---
        st.write("**Add a new user:**")
        new_username = st.text_input("New username")
        new_name = st.text_input("Full name")
        new_password = st.text_input("Password", type="password")
        new_role = st.selectbox("Role", ["admin", "auditor", "viewer"])

        if st.button("Add user"):
            if new_username and new_name and new_password:
                if new_username in users:
                    st.error("That username already exists.")
                else:
                    # hash the new password (newer streamlit-authenticator syntax)
                    hashed = stauth.Hasher.hash_list([new_password])[0]
                    config["credentials"]["usernames"][new_username] = {
                        "name": new_name,
                        "password": hashed,
                        "roles": new_role,
                    }
                    with open("config.yaml", "w") as file:
                        yaml.dump(config, file, default_flow_style=False)
                    st.success(f"User '{new_username}' added as {new_role}. "
                               f"Refresh to see them in the list.")
            else:
                st.warning("Please fill in username, name, and password.")

        st.divider()

        # --- remove a user ---
        st.write("**Remove a user:**")
        removable = [u for u in users if u != username]  # can't remove yourself
        if removable:
            to_remove = st.selectbox("Select a user to remove", removable)
            if st.button("Remove selected user"):
                del config["credentials"]["usernames"][to_remove]
                with open("config.yaml", "w") as file:
                    yaml.dump(config, file, default_flow_style=False)
                st.success(f"User '{to_remove}' removed. Refresh to update the list.")
        else:
            st.caption("No other users to remove.")

        st.caption("This panel is only visible to admin users.")
        st.divider()

        # --- edit an existing user (name and/or password) ---
        st.write("**Edit a user:**")
        editable = list(users.keys())
        edit_user = st.selectbox("Select a user to edit", editable, key="edit_select")

        # pre-fill with their current name
        current_name = users[edit_user]["name"]
        edited_name = st.text_input("New full name (leave as-is to keep)", value=current_name, key="edit_name")
        edited_password = st.text_input("New password (leave blank to keep current)", type="password", key="edit_pw")
        edited_role = st.selectbox("Role", ["admin", "auditor", "viewer"],
                                   index=["admin", "auditor", "viewer"].index(users[edit_user]["roles"]),
                                   key="edit_role")

        if st.button("Save changes"):
            # update name
            config["credentials"]["usernames"][edit_user]["name"] = edited_name
            # update role
            config["credentials"]["usernames"][edit_user]["roles"] = edited_role
            # update password only if a new one was typed
            if edited_password.strip():
                new_hashed = stauth.Hasher.hash_list([edited_password])[0]
                config["credentials"]["usernames"][edit_user]["password"] = new_hashed
            # save to file
            with open("config.yaml", "w") as file:
                yaml.dump(config, file, default_flow_style=False)
            st.success(f"User '{edit_user}' updated. Refresh to see changes.")
