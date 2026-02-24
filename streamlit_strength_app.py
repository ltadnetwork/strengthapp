import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
import io
import tempfile
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Path to logo image (place logo.png alongside this script)
logo_path = "logo.png"

# Configure page and header
st.set_page_config(page_title="Youth Athlete Training System - Strength Assessment App", layout="wide")
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=100)
with col_title:
    st.title("Youth Athlete Training System - Strength Assessment App")

# Scoring thresholds (for Tab 2)
thresholds = {
    'Full Squat':             {'M':{'5':40,'4':33,'3':26,'2':19,'1':11,'0':0}, 'F':{'5':40,'4':33,'3':26,'2':19,'1':11,'0':0}},
    'Pull-Up':                {'M':{'5':12,'4':9,'3':6,'2':4,'1':1,'0':0},    'F':{'5':6,'4':4,'3':3,'2':2,'1':1,'0':0}},
    'Push-Up':                {'M':{'5':35,'4':28,'3':20,'2':10,'1':3,'0':0},  'F':{'5':20,'4':15,'3':10,'2':5,'1':1,'0':0}},
    'Single-Leg Squat Left':  {'M':{'5':5,'4':4,'3':3,'2':2,'1':1,'0':0},     'F':{'5':5,'4':4,'3':3,'2':2,'1':1,'0':0}},
    'Single-Leg Squat Right': {'M':{'5':5,'4':4,'3':3,'2':2,'1':1,'0':0},     'F':{'5':5,'4':4,'3':3,'2':2,'1':1,'0':0}},
    'Front Plank':            {'M':{'5':121,'4':90,'3':60,'2':30,'1':10,'0':0},'F':{'5':121,'4':90,'3':60,'2':30,'1':10,'0':0}},
    'Twisting Sit-Up':        {'M':{'5':40,'4':33,'3':26,'2':19,'1':11,'0':0}, 'F':{'5':40,'4':33,'3':26,'2':19,'1':11,'0':0}}
}

def score_value(exercise, val, sex):
    th = thresholds[exercise][sex]
    for p in range(5, -1, -1):
        if val >= th[str(p)]:
            return p
    return 0

@st.cache_data
def compute_tab1(vals):
    df1 = pd.DataFrame({
        "Exercise": ["Bodyweight Squat","Push-Up","Bodyweight Lunge","Inverted Row","Plank","Side Plank"],
        "Technique": vals
    })
    fig1 = go.Figure()
    fig1.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=df1['Exercise'].tolist() + [df1['Exercise'][0]],
        fill='toself', mode='lines+markers'
    ))
    fig1.update_layout(polar=dict(radialaxis=dict(range=[0,5])), showlegend=False)
    return df1, fig1

@st.cache_data
def compute_tab2(reps, sex):
    df2 = pd.DataFrame({"Exercise": list(thresholds.keys()), "Value": reps})
    df2['Score'] = df2.apply(lambda r: score_value(r['Exercise'], r['Value'], sex), axis=1)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatterpolar(
        r=df2['Score'].tolist() + [df2['Score'][0]],
        theta=df2['Exercise'].tolist() + [df2['Exercise'][0]],
        fill='toself', mode='lines+markers'
    ))
    fig2.update_layout(polar=dict(radialaxis=dict(range=[0,5])), showlegend=False)
    return df2, fig2

@st.cache_data
def compute_tab3(params):
    bw, reps_s, load_s, reps_b, load_b, reps_d, load_d, reps_pc, load_pc, reps_pu, load_pu = params
    est_s  = load_s * (1 + reps_s * 0.0333)
    est_b  = load_b * (1 + reps_b * 0.0333)
    est_d  = load_d * (1 + reps_d * 0.0333)
    est_pc = load_pc * (1 + reps_pc * 0.0333)
    total_load_pu = bw + load_pu
    est_total_pu  = total_load_pu * (1 + reps_pu * 0.0333)
    est_pu = est_total_pu - bw
    lifts = ["Squat","Bench","Deadlift","Power Clean","Pull-Up"]
    loads = [round(load_s,1), round(load_b,1), round(load_d,1), round(load_pc,1), round(load_pu,1)]
    reps_list = [reps_s, reps_b, reps_d, reps_pc, reps_pu]
    ests_rounded = [round(e,1) for e in [est_s, est_b, est_d, est_pc, est_pu]]
    ratios = [round(e/bw,2) if bw else 0 for e in [est_s, est_b, est_d, est_pc, est_pu]]
    df3 = pd.DataFrame({
        "Lift": lifts,
        "Load (kg)": loads,
        "Reps": reps_list,
        "Est. 1RM (kg)": ests_rounded,
        "1RM/BW": ratios
    })
    fig3 = go.Figure()
    fig3.add_trace(go.Scatterpolar(
        r=ratios + [ratios[0]],
        theta=lifts + [lifts[0]],
        fill='toself', mode='lines+markers'
    ))
    fig3.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
    return df3, fig3

# Radar chart using matplotlib - no kaleido needed
def make_radar_png(labels, values, max_val=5):
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals = list(values) + [values[0]]
    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, linewidth=2, linestyle='solid', color='steelblue')
    ax.fill(angles, vals, alpha=0.25, color='steelblue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8)
    ax.set_ylim(0, max_val)
    ax.yaxis.set_tick_params(labelsize=7)
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(tmp.name, bbox_inches='tight', dpi=150)
    plt.close(fig)
    tmp.close()
    return tmp.name

# PDF generation
def create_pdf(title, info: dict, df: pd.DataFrame, chart_labels=None, chart_values=None, max_val=5):
    pdf = FPDF()
    pdf.add_page()
    try:
        logo_w = 30
        pdf.image(logo_path, x=pdf.w - pdf.r_margin - logo_w, y=pdf.t_margin, w=logo_w)
    except Exception:
        pass
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", '', 12)
    for k, v in info.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)
    pdf.ln(5)
    if chart_labels and chart_values:
        tmp_path = None
        try:
            tmp_path = make_radar_png(chart_labels, chart_values, max_val)
            w = pdf.w - 2 * pdf.l_margin
            pdf.image(tmp_path, x=pdf.l_margin, y=pdf.get_y(), w=w)
            pdf.ln(w * 0.75)
        except Exception:
            pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
    pdf.set_font("Arial", 'B', 12)
    ew = pdf.w - 2 * pdf.l_margin
    cw = ew / len(df.columns)
    for col in df.columns:
        pdf.cell(cw, 8, str(col), border=1, align='C')
    pdf.ln()
    pdf.set_font("Arial", '', 12)
    for _, row in df.iterrows():
        for item in row:
            pdf.cell(cw, 8, str(item), border=1, align='C')
        pdf.ln()
    raw = pdf.output(dest='S')
    if isinstance(raw, (bytes, bytearray)):
        return io.BytesIO(raw)
    return io.BytesIO(raw.encode('latin-1'))

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Movement Competency", "Bodyweight Strength Test", "Relative Strength Assessment"])

# Tab 1: Movement Competency
with tabs[0]:
    if 'calc1' not in st.session_state:
        st.session_state.calc1 = False
    c1, c2 = st.columns([1, 2])
    with c1:
        name = st.text_input("Athlete Name")
        test_date = st.date_input("Test Date")
        age = st.number_input("Age (years)", min_value=0, value=18)
        pah = st.number_input("% Predicted Adult Height", min_value=0, value=100)
        vals = [
            st.slider("Bodyweight Squat Tech (1-5)", 1, 5, 3),
            st.slider("Push-Up Tech (1-5)", 1, 5, 3),
            st.slider("Bodyweight Lunge Tech (1-5)", 1, 5, 3),
            st.slider("Inverted Row Tech (1-5)", 1, 5, 3),
            st.slider("Plank Tech (1-5)", 1, 5, 3),
            st.slider("Side Plank Tech (1-5)", 1, 5, 3)
        ]
        if st.button("Calculate Movement Competency"):
            st.session_state.calc1 = True
    with c2:
        if st.session_state.calc1:
            df1, fig1 = compute_tab1(vals)
            st.plotly_chart(fig1, use_container_width=True)
            st.table(df1)
            info1 = {"Athlete": name, "Test Date": test_date.strftime('%Y-%m-%d'), "Age": f"{age} yrs", "% PAH": f"{pah}%"}
            labels1 = ["Bodyweight Squat","Push-Up","Bodyweight Lunge","Inverted Row","Plank","Side Plank"]
            pdf1 = create_pdf("Movement Competency Report", info1, df1, chart_labels=labels1, chart_values=vals, max_val=5)
            st.download_button("Download Movement Competency Report", pdf1, "movement_competency_report.pdf", "application/pdf")
    _, col_logo1 = st.columns([10,1])
    with col_logo1:
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)

# Tab 2: Bodyweight Strength Test
with tabs[1]:
    if 'calc2' not in st.session_state:
        st.session_state.calc2 = False
    c1, c2 = st.columns([1, 2])
    with c1:
        name2 = st.text_input("Name", key="bw_name")
        test_date2 = st.date_input("Test Date", key="bw_date")
        sex = st.selectbox("Sex", ["M", "F"], format_func=lambda x: "Male" if x == 'M' else "Female")
        reps = [
            st.number_input("Squat reps (60s @10% BW)", 0, 500, 0),
            st.number_input("Pull-Up reps", 0, 500, 0),
            st.number_input("Push-Up reps", 0, 500, 0),
            st.number_input("Single-Leg Left reps", 0, 100, 0),
            st.number_input("Single-Leg Right reps", 0, 100, 0),
            st.number_input("Front Plank secs", 0, 1000, 0),
            st.number_input("Twist Sit-Up reps", 0, 500, 0)
        ]
        if st.button("Calculate Bodyweight Strength Test"):
            st.session_state.calc2 = True
    with c2:
        if st.session_state.calc2:
            df2, fig2 = compute_tab2(reps, sex)
            st.plotly_chart(fig2, use_container_width=True)
            st.table(df2)
            avg_leg = (df2.loc[3, 'Score'] + df2.loc[4, 'Score']) / 2
            total2 = df2.loc[[0,1,2,5,6], 'Score'].sum() + avg_leg
            result2 = "Pass" if total2 > 18 else "Fail"
            st.table(pd.DataFrame({"Total Score": [round(total2,1)], "Result": [result2]}))
            info2 = {"Name": name2, "Test Date": test_date2.strftime('%Y-%m-%d'), "Sex": sex}
            labels2 = list(thresholds.keys())
            scores2 = df2['Score'].tolist()
            pdf2 = create_pdf("Bodyweight Strength Test Report", info2, df2, chart_labels=labels2, chart_values=scores2, max_val=5)
            st.download_button("Download Bodyweight Strength Test Report", pdf2, "bodyweight_strength_test_report.pdf", "application/pdf")
    _, col_logo2 = st.columns([10,1])
    with col_logo2:
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)

# Tab 3: Relative Strength Assessment
with tabs[2]:
    c1, c2 = st.columns([1, 2])
    with c1:
        name3 = st.text_input("Athlete Name", key="rs_name")
        test_date3 = st.date_input("Test Date", key="rs_date")
        bw = st.number_input("Body Mass (kg)", 0.0, 300.0, 70.0, step=0.1)
        reps_s = st.number_input("Squat reps", 0, 20, 5)
        load_s = st.number_input("Squat load (kg)", 0.0, 500.0, 100.0)
        reps_b = st.number_input("Bench reps", 0, 20, 5)
        load_b = st.number_input("Bench load (kg)", 0.0, 500.0, 60.0)
        reps_d = st.number_input("Deadlift reps", 0, 20, 5)
        load_d = st.number_input("Deadlift load (kg)", 0.0, 500.0, 140.0)
        reps_pc = st.number_input("Power Clean reps", 0, 20, 5)
        load_pc = st.number_input("Power Clean load (kg)", 0.0, 500.0, 80.0)
        reps_pu = st.number_input("Pull-Up reps", 0, 50, 5)
        load_pu = st.number_input("Pull up additional load (kg)", 0.0, 100.0, 0.0)
        calculate3 = st.button("Calculate Relative Strength")
    with c2:
        if calculate3:
            params = (bw, reps_s, load_s, reps_b, load_b, reps_d, load_d, reps_pc, load_pc, reps_pu, load_pu)
            df3, fig3 = compute_tab3(params)
            st.plotly_chart(fig3, use_container_width=True)
            st.table(df3)
            info3 = {"Athlete": name3, "Test Date": test_date3.strftime('%Y-%m-%d'), "Body Mass": f"{bw} kg"}
            labels3 = ["Squat","Bench","Deadlift","Power Clean","Pull-Up"]
            ratios3 = df3['1RM/BW'].tolist()
            max_val3 = max(ratios3) * 1.2 if any(r > 0 for r in ratios3) else 3
            pdf3 = create_pdf("Relative Strength Report", info3, df3, chart_labels=labels3, chart_values=ratios3, max_val=max_val3)
            st.download_button("Download Relative Strength Report", pdf3, "relative_strength_report.pdf", "application/pdf")
    _, col_logo3 = st.columns([10,1])
    with col_logo3:
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)
