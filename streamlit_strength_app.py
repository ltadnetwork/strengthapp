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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Youth Strength Training Assessment",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Design tokens (matching index.html) ───────────────────────────────────────
NAVY_DARK  = "#080F1E"
NAVY       = "#0F1B34"
NAVY_MID   = "#1A2B4A"
NAVY_LIGHT = "#253A5E"
GREEN      = "#23FF00"
GREY       = "#9F9F9F"
FOUND      = "#1D9E75"
DEV        = "#7F77DD"
PERF       = "#E8622A"

# ── Global CSS injection ──────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@400;500;600&display=swap" rel="stylesheet">

<style>
/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

:root {
  --navy:       #0F1B34;
  --navy-mid:   #1A2B4A;
  --navy-light: #253A5E;
  --navy-dark:  #080F1E;
  --green:      #23FF00;
  --green-dim:  rgba(35,255,0,0.12);
  --grey:       #9F9F9F;
  --found:      #1D9E75;
  --found-bg:   rgba(29,158,117,0.12);
  --found-dim:  rgba(29,158,117,0.25);
  --dev:        #7F77DD;
  --dev-bg:     rgba(127,119,221,0.12);
  --dev-dim:    rgba(127,119,221,0.25);
  --perf:       #E8622A;
  --perf-bg:    rgba(232,98,42,0.12);
  --perf-dim:   rgba(232,98,42,0.25);
  --border:     rgba(255,255,255,0.07);
  --row-a:      rgba(255,255,255,0.025);
  --row-b:      rgba(255,255,255,0.055);
  --radius:     10px;
}

/* ── App shell ── */
.stApp {
  background: var(--navy-dark) !important;
  font-family: 'Barlow', sans-serif !important;
  color: #FFFFFF !important;
}

/* Background glow texture */
.stApp::before {
  content: '';
  position: fixed; inset: 0; z-index: 0;
  background:
    radial-gradient(ellipse 60% 50% at 80% 10%, rgba(35,255,0,0.04) 0%, transparent 70%),
    radial-gradient(ellipse 40% 60% at 5% 80%,  rgba(35,255,0,0.03) 0%, transparent 60%);
  pointer-events: none;
}

/* Hide default Streamlit chrome */
#MainMenu, header[data-testid="stHeader"], footer { display: none !important; }
.block-container { padding-top: 0 !important; padding-bottom: 60px !important; max-width: 1100px !important; }

/* ── All text elements ── */
.stApp p, .stApp label, .stApp div, .stApp span,
.stApp .stMarkdown, .stApp .stText { color: #FFFFFF; font-family: 'Barlow', sans-serif; }

/* ── Inputs ── */
.stApp .stTextInput input,
.stApp .stNumberInput input,
.stApp .stDateInput input,
.stApp .stSelectbox > div > div {
  background: var(--navy-mid) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: #FFFFFF !important;
  font-family: 'Barlow', sans-serif !important;
  font-size: 14px !important;
}

.stApp .stTextInput label,
.stApp .stNumberInput label,
.stApp .stDateInput label,
.stApp .stSelectbox label,
.stApp .stSlider label {
  color: var(--grey) !important;
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
  background: var(--green) !important;
  border-color: var(--green) !important;
}
/* Hide the min/max tick label row — target every known Streamlit selector */
[data-testid="stSlider"] div[data-testid="stTickBar"],
[data-testid="stSlider"] [data-testid="stTickBarItem"],
[data-testid="stSlider"] div[class*="tickBar"],
[data-testid="stSlider"] div[class*="TickBar"],
[data-testid="stSlider"] > div > div > div > div:last-child > div > div,
[data-testid="stSlider"] > label + div > div > div:nth-child(3),
[data-testid="stSlider"] > label + div > div > div:last-child {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  overflow: hidden !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  gap: 8px !important;
  border-bottom: 1.5px solid var(--border) !important;
  padding-bottom: 0 !important;
}

.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border: 1.5px solid transparent !important;
  border-bottom: none !important;
  border-radius: 8px 8px 0 0 !important;
  color: var(--grey) !important;
  font-family: 'Barlow Condensed', sans-serif !important;
  font-weight: 700 !important;
  font-size: 13px !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  padding: 8px 20px !important;
  transition: all 0.18s ease !important;
}

.stTabs [aria-selected="true"] {
  background: var(--green-dim) !important;
  border-color: var(--green) !important;
  border-bottom-color: var(--navy-dark) !important;
  color: var(--green) !important;
}

.stTabs [data-baseweb="tab"]:hover {
  color: #FFFFFF !important;
  border-color: var(--border) !important;
}

.stTabs [data-baseweb="tab-panel"] {
  padding: 24px 0 0 !important;
  background: transparent !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--green-dim) !important;
  border: 1.5px solid var(--green) !important;
  border-radius: 8px !important;
  color: var(--green) !important;
  font-family: 'Barlow Condensed', sans-serif !important;
  font-weight: 700 !important;
  font-size: 13px !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  padding: 8px 20px !important;
  transition: all 0.18s !important;
}
.stButton > button:hover {
  background: var(--green) !important;
  color: var(--navy-dark) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
  background: var(--green-dim) !important;
  border: 1.5px solid var(--green) !important;
  border-radius: 8px !important;
  color: var(--green) !important;
  font-family: 'Barlow Condensed', sans-serif !important;
  font-weight: 700 !important;
  font-size: 12px !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
}
.stDownloadButton > button:hover {
  background: var(--green) !important;
  color: var(--navy-dark) !important;
}

/* ── Dataframe / Table ── */
.stDataFrame, .stTable { background: transparent !important; }
.stDataFrame table, .stTable table {
  border-collapse: collapse !important;
  width: 100% !important;
  font-family: 'Barlow', sans-serif !important;
  font-size: 13px !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
}
.stDataFrame thead th, .stTable thead th {
  background: var(--navy-mid) !important;
  color: var(--grey) !important;
  font-family: 'Barlow Condensed', sans-serif !important;
  font-weight: 700 !important;
  font-size: 12px !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  padding: 10px 14px !important;
  border-bottom: 1px solid var(--border) !important;
}
.stDataFrame tbody tr:nth-child(odd), .stTable tbody tr:nth-child(odd) {
  background: var(--row-a) !important;
}
.stDataFrame tbody tr:nth-child(even), .stTable tbody tr:nth-child(even) {
  background: var(--row-b) !important;
}
.stDataFrame tbody td, .stTable tbody td {
  padding: 10px 14px !important;
  border-top: 1px solid var(--border) !important;
  color: rgba(255,255,255,0.9) !important;
}

/* ── Selectbox dropdown ── */
[data-baseweb="select"] {
  background: var(--navy-mid) !important;
}
[data-baseweb="popover"] {
  background: var(--navy-mid) !important;
  border: 1px solid var(--border) !important;
}
[data-baseweb="menu"] { background: var(--navy-mid) !important; }
[role="option"] { background: var(--navy-mid) !important; color: #FFF !important; }
[role="option"]:hover { background: var(--navy-light) !important; }

/* ── Section card ── */
.section-card {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--row-a);
  padding: 20px 24px;
  margin-bottom: 20px;
}

/* ── Input panel ── */
.input-panel {
  background: var(--navy-mid);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 18px;
}

/* ── Metric badge ── */
.metric-badge {
  background: var(--navy-mid);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
  text-align: center;
}
.metric-badge .metric-val {
  font-family: 'Barlow Condensed', sans-serif;
  font-weight: 800; font-size: 28px;
  line-height: 1;
}
.metric-badge .metric-label {
  font-size: 10px; letter-spacing: 0.1em;
  color: var(--grey); text-transform: uppercase;
  margin-top: 4px;
}

/* ── Pass / Fail pill ── */
.result-pill {
  display: inline-block;
  font-family: 'Barlow Condensed', sans-serif;
  font-weight: 800; font-size: 16px; letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 6px 20px; border-radius: 20px;
}
.result-pass {
  background: rgba(29,158,117,0.18);
  border: 1.5px solid var(--found);
  color: var(--found);
}
.result-fail {
  background: rgba(232,98,42,0.18);
  border: 1.5px solid var(--perf);
  color: var(--perf);
}

/* ── Divider ── */
.green-rule {
  height: 1.5px;
  background: linear-gradient(90deg, var(--green) 0%, rgba(35,255,0,0.15) 60%, transparent 100%);
  margin: 20px 0 28px;
}

/* ── Number input spinners ── */
.stNumberInput button {
  background: var(--navy-light) !important;
  border-color: var(--border) !important;
  color: #FFF !important;
}

/* ── Hide slider tick/None labels only ── */
.stSlider [data-testid="stTickBar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# JS: remove any element whose sole visible text content is "None"
st.markdown("""
<script>
(function removeNonePills() {
  function clean() {
    document.querySelectorAll('div[data-testid="stSlider"] *').forEach(el => {
      if (el.children.length === 0 && el.textContent.trim() === 'None') {
        el.style.display = 'none';
      }
    });
  }
  // Run immediately and after short delays to catch deferred renders
  clean();
  setTimeout(clean, 300);
  setTimeout(clean, 800);
  setTimeout(clean, 2000);
  // Also watch for DOM mutations
  new MutationObserver(clean).observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)

# ── Scoring thresholds ────────────────────────────────────────────────────────
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

# ── Logo path ─────────────────────────────────────────────────────────────────
logo_path = "logo.png"

# ── Plotly theme ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Barlow, sans-serif", color="#FFFFFF"),
    showlegend=False,
    height=480,
    margin=dict(l=40, r=40, t=40, b=40),
)
PLOTLY_POLAR = dict(
    radialaxis=dict(
        gridcolor="rgba(255,255,255,0.08)",
        linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color=GREY, size=10),
    ),
    angularaxis=dict(
        tickfont=dict(size=13, family="Barlow Condensed, sans-serif", color="#FFFFFF"),
        linecolor="rgba(255,255,255,0.1)",
        gridcolor="rgba(255,255,255,0.06)",
    ),
    bgcolor="rgba(0,0,0,0)",
)

FILL_COLORS = {
    "#23FF00": "rgba(35,255,0,0.18)",
    "#7F77DD": "rgba(127,119,221,0.18)",
    "#E8622A": "rgba(232,98,42,0.18)",
}

def make_radar(labels, values, color, max_val=5):
    fill_color = FILL_COLORS.get(color, "rgba(35,255,0,0.18)")
    vals = list(values) + [values[0]]
    lbls = list(labels) + [labels[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=lbls,
        fill='toself',
        mode='lines+markers',
        line=dict(color=color, width=2.5),
        fillcolor=fill_color,
        marker=dict(color=color, size=7),
    ))
    layout = dict(**PLOTLY_LAYOUT)
    layout['polar'] = dict(**PLOTLY_POLAR)
    layout['polar']['radialaxis']['range'] = [0, max_val]
    fig.update_layout(**layout)
    return fig



# ── PDF generation ────────────────────────────────────────────────────────────
def safe_str(text):
    """Encode text as latin-1 safe — replaces any character FPDF cannot handle."""
    return str(text).encode('latin-1', errors='replace').decode('latin-1')

def make_radar_png(labels, values, max_val=5):
    """Render a high-quality perfectly circular radar chart PNG for the PDF."""
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals = list(values) + [values[0]]

    # Use a square figure with generous padding so labels aren't clipped
    fig = plt.figure(figsize=(7, 7), facecolor='#FFFFFF')
    # Place axes as a square in the centre of the figure
    ax = fig.add_axes([0.15, 0.15, 0.70, 0.70], polar=True)
    ax.set_facecolor('#F7F9FC')

    # Fill & line
    ax.fill(angles, vals, alpha=0.20, color='#1D9E75')
    ax.plot(angles, vals, linewidth=2.5, linestyle='solid', color='#1D9E75')
    ax.scatter(angles[:-1], vals[:-1], s=70, color='#1D9E75', zorder=5)

    # Axis styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10, color='#0F1B34', fontweight='bold')
    ax.tick_params(axis='x', pad=16)
    ax.set_ylim(0, max_val)
    ax.yaxis.set_tick_params(labelsize=8, colors='#9F9F9F')
    ax.spines['polar'].set_color('#CCCCCC')
    ax.grid(color='#DDDDDD', linewidth=0.8)

    # Save at exact figure size — do NOT use bbox_inches='tight' as it
    # can introduce asymmetric cropping that makes the circle look oval
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(tmp.name, dpi=300, facecolor='#FFFFFF')
    plt.close(fig)
    tmp.close()
    return tmp.name

def create_pdf(title, info: dict, df: pd.DataFrame, chart_labels=None, chart_values=None, max_val=5):
    pdf = FPDF()
    pdf.add_page()

    # Logo (top-right)
    try:
        logo_w = 28
        pdf.image(logo_path, x=pdf.w - pdf.r_margin - logo_w, y=pdf.t_margin, w=logo_w)
    except Exception:
        pass

    # Title
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(35, 180, 0)
    pdf.cell(0, 12, safe_str(title), ln=True, align='C')
    pdf.ln(4)

    # Info block
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(80, 80, 80)
    for k, v in info.items():
        pdf.cell(0, 7, safe_str(f"{k}: {v}"), ln=True)
    pdf.ln(6)

    # Radar chart — sized to fit cleanly on the page
    if chart_labels and chart_values:
        tmp_path = None
        try:
            tmp_path = make_radar_png(chart_labels, chart_values, max_val)
            # Use a fixed square size that fits nicely: 120mm
            img_size = 120
            x_pos = (pdf.w - img_size) / 2   # centre horizontally
            y_before = pdf.get_y()
            pdf.image(tmp_path, x=x_pos, y=y_before, w=img_size, h=img_size)
            pdf.set_y(y_before + img_size + 6)  # move cursor below image
        except Exception as e:
            pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Table header — dark text on light background
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(20, 30, 60)
    pdf.set_fill_color(230, 235, 245)
    ew = pdf.w - 2 * pdf.l_margin
    cw = ew / len(df.columns)
    for col in df.columns:
        pdf.cell(cw, 9, safe_str(col), border=1, align='C', fill=True)
    pdf.ln()

    # Table rows — dark text on white / light-grey alternating rows
    pdf.set_font("Arial", '', 11)
    for i, (_, row) in enumerate(df.iterrows()):
        fill = i % 2 == 0
        pdf.set_fill_color(247, 249, 252) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(20, 30, 60)
        for item in row:
            pdf.cell(cw, 8, safe_str(item), border=1, align='C', fill=True)
        pdf.ln()

    raw = pdf.output(dest='S')
    if isinstance(raw, (bytes, bytearray)):
        return io.BytesIO(raw)
    return io.BytesIO(raw.encode('latin-1'))

# ── Cached computations ───────────────────────────────────────────────────────
@st.cache_data
def compute_tab1(vals):
    df1 = pd.DataFrame({
        "Exercise": ["Squat","Push-Up","Lunge","Inverted Row","Plank","Side Plank"],
        "Technique Score": vals
    })
    return df1

@st.cache_data
def compute_tab2(reps, sex):
    df2 = pd.DataFrame({"Exercise": list(thresholds.keys()), "Value": reps})
    df2['Score'] = df2.apply(lambda r: score_value(r['Exercise'], r['Value'], sex), axis=1)
    return df2

@st.cache_data
def compute_tab3(params):
    bw, reps_s, load_s, reps_b, load_b, reps_d, load_d, reps_pc, load_pc, reps_pu, load_pu = params
    est_s  = load_s  * (1 + reps_s  * 0.0333)
    est_b  = load_b  * (1 + reps_b  * 0.0333)
    est_d  = load_d  * (1 + reps_d  * 0.0333)
    est_pc = load_pc * (1 + reps_pc * 0.0333)
    total_load_pu = bw + load_pu
    est_total_pu  = total_load_pu * (1 + reps_pu * 0.0333)
    est_pu = est_total_pu - bw
    lifts = ["Squat","Bench","Deadlift","Power Clean","Pull-Up"]
    loads = [round(load_s,1), round(load_b,1), round(load_d,1), round(load_pc,1), round(load_pu,1)]
    reps_list = [reps_s, reps_b, reps_d, reps_pc, reps_pu]
    ests = [round(e,1) for e in [est_s, est_b, est_d, est_pc, est_pu]]
    ratios = [round(e/bw,2) if bw else 0 for e in [est_s, est_b, est_d, est_pc, est_pu]]
    df3 = pd.DataFrame({
        "Lift": lifts,
        "Load (kg)": loads,
        "Reps": reps_list,
        "Est. 1RM (kg)": ests,
        "1RM / BW": ratios
    })
    return df3

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between;
            padding:28px 0 0; flex-wrap:wrap; gap:16px;">
  <div style="display:flex; align-items:center; gap:16px;">
    <div style="width:48px;height:48px;flex-shrink:0;">
      <svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
        <rect width="48" height="48" rx="10" fill="#1A2B4A"/>
        <rect x="6" y="22" width="36" height="4" rx="2" fill="#23FF00"/>
        <rect x="6" y="13" width="8" height="22" rx="2" fill="#23FF00"/>
        <rect x="34" y="13" width="8" height="22" rx="2" fill="#23FF00"/>
        <rect x="3" y="19" width="5" height="10" rx="2" fill="rgba(35,255,0,0.45)"/>
        <rect x="40" y="19" width="5" height="10" rx="2" fill="rgba(35,255,0,0.45)"/>
      </svg>
    </div>
    <div>
      <div style="font-family:'Barlow Condensed',sans-serif;font-weight:800;
                  font-size:22px;letter-spacing:0.06em;color:#FFFFFF;line-height:1;">
        LTAD NETWORK
      </div>
      <div style="font-size:10px;letter-spacing:0.12em;color:#23FF00;
                  text-transform:uppercase;margin-top:2px;">
        Strength &amp; Conditioning
      </div>
    </div>
  </div>
  <div style="text-align:right;">
    <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                font-size:18px;color:#FFFFFF;letter-spacing:0.03em;">
      Youth Strength Assessment App
    </div>
    <div style="font-size:11px;color:#9F9F9F;margin-top:2px;">
      Movement · Bodyweight · Relative Strength
    </div>
  </div>
</div>
<div class="green-rule" style="height:1.5px;
  background:linear-gradient(90deg,#23FF00 0%,rgba(35,255,0,0.15) 60%,transparent 100%);
  margin:20px 0 28px;"></div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "01 · Movement Competency",
    "02 · Bodyweight Strength",
    "03 · Relative Strength"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Movement Competency
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div style="margin-bottom:20px;">
      <div style="display:flex;align-items:center;gap:12px;padding:11px 16px;
                  background:#1A2B4A;border-radius:10px 10px 0 0;
                  border:1px solid rgba(255,255,255,0.07);border-bottom:none;">
        <span style="font-family:'Barlow Condensed',sans-serif;font-weight:800;
                     font-size:11px;letter-spacing:0.12em;color:#23FF00;
                     text-transform:uppercase;background:rgba(35,255,0,0.12);
                     padding:2px 8px;border-radius:4px;">01</span>
        <span style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                     font-size:14px;letter-spacing:0.06em;text-transform:uppercase;
                     color:#FFFFFF;">Movement Competency Assessment</span>
      </div>
      <div style="padding:14px 16px;background:rgba(255,255,255,0.025);
                  border:1px solid rgba(255,255,255,0.07);border-top:none;
                  border-radius:0 0 10px 10px;font-size:13px;color:rgba(255,255,255,0.7);">
        Rate each fundamental movement pattern on a 1–5 technical competency scale.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1.6], gap="large")

    with col_in:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                    font-size:12px;letter-spacing:0.1em;text-transform:uppercase;
                    color:#9F9F9F;margin-bottom:16px;">Athlete Details</div>""",
                    unsafe_allow_html=True)
        name = st.text_input("Athlete Name", placeholder="e.g. Alex Johnson", key="mc_name")
        col_a, col_b = st.columns(2)
        with col_a:
            test_date = st.date_input("Test Date", key="mc_date")
        with col_b:
            age = st.number_input("Age (yrs)", min_value=0, value=16, key="mc_age")
        pah = st.number_input("% Predicted Adult Height", min_value=0, max_value=100, value=95, key="mc_pah")

        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                    font-size:12px;letter-spacing:0.1em;text-transform:uppercase;
                    color:#9F9F9F;margin:20px 0 12px;">Movement Scores</div>""",
                    unsafe_allow_html=True)
        exercises_t1 = ["Squat", "Push-Up", "Lunge", "Inverted Row", "Plank", "Side Plank"]
        vals = []
        for ex in exercises_t1:
            v = st.slider(ex, min_value=1, max_value=5, value=3, key=f"mc_{ex}")
            vals.append(v)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_out:
        df1 = compute_tab1(tuple(vals))
        fig1 = make_radar(exercises_t1, vals, GREEN)
        st.plotly_chart(fig1, use_container_width=True)

        # Scores table
        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                    font-size:12px;letter-spacing:0.1em;text-transform:uppercase;
                    color:#9F9F9F;margin-bottom:8px;">Score Summary</div>""",
                    unsafe_allow_html=True)
        _cols1 = df1.columns.tolist()
        _rows1 = df1.values.tolist()
        _hdr1 = "".join(f'<th style="padding:10px 14px;font-family:\'Barlow Condensed\',sans-serif;font-weight:700;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;color:#9F9F9F;background:#1A2B4A;border-bottom:1px solid rgba(255,255,255,0.07);">{c}</th>' for c in _cols1)
        _body1 = ""
        for _i, _row in enumerate(_rows1):
            _bg = "rgba(255,255,255,0.025)" if _i % 2 == 0 else "rgba(255,255,255,0.055)"
            _cells = "".join(f'<td style="padding:10px 14px;font-size:13px;color:rgba(255,255,255,0.9);border-top:1px solid rgba(255,255,255,0.07);">{v}</td>' for v in _row)
            _body1 += f'<tr style="background:{_bg}">{_cells}</tr>'
        st.markdown(f'''<table style="width:100%;border-collapse:collapse;border-radius:10px;overflow:hidden;border:1px solid rgba(255,255,255,0.07);">
          <thead><tr>{_hdr1}</tr></thead><tbody>{_body1}</tbody></table>''', unsafe_allow_html=True)

        # Download
        info1 = {
            "Athlete": name or "—",
            "Test Date": test_date.strftime('%d %b %Y'),
            "Age": f"{age} yrs",
            "% PAH": f"{pah}%"
        }
        pdf1 = create_pdf("Movement Competency Report", info1, df1,
                          chart_labels=exercises_t1, chart_values=vals, max_val=5)
        st.download_button("⬇ Download PDF Report", pdf1,
                           "movement_competency_report.pdf", "application/pdf",
                           key="dl_mc")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Bodyweight Strength Test
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div style="margin-bottom:20px;">
      <div style="display:flex;align-items:center;gap:12px;padding:11px 16px;
                  background:#1A2B4A;border-radius:10px 10px 0 0;
                  border:1px solid rgba(255,255,255,0.07);border-bottom:none;">
        <span style="font-family:'Barlow Condensed',sans-serif;font-weight:800;
                     font-size:11px;letter-spacing:0.12em;color:#23FF00;
                     text-transform:uppercase;background:rgba(35,255,0,0.12);
                     padding:2px 8px;border-radius:4px;">02</span>
        <span style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                     font-size:14px;letter-spacing:0.06em;text-transform:uppercase;
                     color:#FFFFFF;">Bodyweight Strength Test</span>
      </div>
      <div style="padding:14px 16px;background:rgba(255,255,255,0.025);
                  border:1px solid rgba(255,255,255,0.07);border-top:none;
                  border-radius:0 0 10px 10px;font-size:13px;color:rgba(255,255,255,0.7);">
        Enter reps / time for each test. Scores are benchmarked by sex on a 0–5 scale. Pass threshold: total score &gt; 18.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_in2, col_out2 = st.columns([1, 1.6], gap="large")

    with col_in2:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                    font-size:12px;letter-spacing:0.1em;text-transform:uppercase;
                    color:#9F9F9F;margin-bottom:16px;">Athlete Details</div>""",
                    unsafe_allow_html=True)
        name2 = st.text_input("Athlete Name", placeholder="e.g. Alex Johnson", key="bw_name")
        col_a2, col_b2 = st.columns(2)
        with col_a2:
            test_date2 = st.date_input("Test Date", key="bw_date")
        with col_b2:
            sex = st.selectbox("Sex", ["M", "F"],
                               format_func=lambda x: "Male" if x == 'M' else "Female",
                               key="bw_sex")

        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                    font-size:12px;letter-spacing:0.1em;text-transform:uppercase;
                    color:#9F9F9F;margin:20px 0 12px;">Test Results</div>""",
                    unsafe_allow_html=True)
        reps = [
            st.number_input("Full Squat reps (60 s @ 10% BW)", 0, 500, 0, key="bw_sq"),
            st.number_input("Pull-Up reps", 0, 500, 0, key="bw_pu"),
            st.number_input("Push-Up reps", 0, 500, 0, key="bw_push"),
            st.number_input("Single-Leg Squat — Left reps", 0, 100, 0, key="bw_sll"),
            st.number_input("Single-Leg Squat — Right reps", 0, 100, 0, key="bw_slr"),
            st.number_input("Front Plank (seconds)", 0, 1000, 0, key="bw_plank"),
            st.number_input("Twisting Sit-Up reps", 0, 500, 0, key="bw_su"),
        ]
        st.markdown('</div>', unsafe_allow_html=True)

    with col_out2:
        df2 = compute_tab2(tuple(reps), sex)
        labels2 = list(thresholds.keys())
        scores2 = df2['Score'].tolist()
        fig2 = make_radar(labels2, scores2, DEV)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                    font-size:12px;letter-spacing:0.1em;text-transform:uppercase;
                    color:#9F9F9F;margin-bottom:8px;">Score Summary</div>""",
                    unsafe_allow_html=True)
        _cols2 = df2.columns.tolist()
        _rows2 = df2.values.tolist()
        _hdr2 = "".join(f'<th style="padding:10px 14px;font-family:\'Barlow Condensed\',sans-serif;font-weight:700;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;color:#9F9F9F;background:#1A2B4A;border-bottom:1px solid rgba(255,255,255,0.07);">{c}</th>' for c in _cols2)
        _body2 = ""
        for _i, _row in enumerate(_rows2):
            _bg = "rgba(255,255,255,0.025)" if _i % 2 == 0 else "rgba(255,255,255,0.055)"
            _cells = "".join(f'<td style="padding:10px 14px;font-size:13px;color:rgba(255,255,255,0.9);border-top:1px solid rgba(255,255,255,0.07);">{v}</td>' for v in _row)
            _body2 += f'<tr style="background:{_bg}">{_cells}</tr>'
        st.markdown(f'''<table style="width:100%;border-collapse:collapse;border-radius:10px;overflow:hidden;border:1px solid rgba(255,255,255,0.07);">
          <thead><tr>{_hdr2}</tr></thead><tbody>{_body2}</tbody></table>''', unsafe_allow_html=True)

        # Total & result
        avg_leg   = (df2.loc[3, 'Score'] + df2.loc[4, 'Score']) / 2
        total2    = df2.loc[[0,1,2,5,6], 'Score'].sum() + avg_leg
        result2   = "PASS" if total2 > 18 else "FAIL"
        pill_cls  = "result-pass" if result2 == "PASS" else "result-fail"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:16px;
                    background:#1A2B4A;border:1px solid rgba(255,255,255,0.07);
                    border-radius:10px;padding:16px 20px;margin:12px 0;">
          <div>
            <div style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;
                        color:#9F9F9F;">Total Score</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-weight:800;
                        font-size:32px;color:#FFFFFF;line-height:1.1;">
              {round(total2,1)}<span style="font-size:16px;color:#9F9F9F;"> / 30</span>
            </div>
          </div>
          <div style="margin-left:8px;">
            <span class="{pill_cls} result-pill">{result2}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        info2 = {
            "Athlete": name2 or "—",
            "Test Date": test_date2.strftime('%d %b %Y'),
            "Sex": "Male" if sex == "M" else "Female",
            "Total Score": f"{round(total2,1)} / 30",
            "Result": result2,
        }
        pdf2 = create_pdf("Bodyweight Strength Test Report", info2, df2,
                          chart_labels=labels2, chart_values=scores2, max_val=5)
        st.download_button("⬇ Download PDF Report", pdf2,
                           "bodyweight_strength_test_report.pdf", "application/pdf",
                           key="dl_bw")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Relative Strength Assessment
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div style="margin-bottom:20px;">
      <div style="display:flex;align-items:center;gap:12px;padding:11px 16px;
                  background:#1A2B4A;border-radius:10px 10px 0 0;
                  border:1px solid rgba(255,255,255,0.07);border-bottom:none;">
        <span style="font-family:'Barlow Condensed',sans-serif;font-weight:800;
                     font-size:11px;letter-spacing:0.12em;color:#23FF00;
                     text-transform:uppercase;background:rgba(35,255,0,0.12);
                     padding:2px 8px;border-radius:4px;">03</span>
        <span style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                     font-size:14px;letter-spacing:0.06em;text-transform:uppercase;
                     color:#FFFFFF;">Relative Strength Assessment</span>
      </div>
      <div style="padding:14px 16px;background:rgba(255,255,255,0.025);
                  border:1px solid rgba(255,255,255,0.07);border-top:none;
                  border-radius:0 0 10px 10px;font-size:13px;color:rgba(255,255,255,0.7);">
        Estimated 1RM calculated via Epley formula. Results expressed as a ratio of body mass.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_in3, col_out3 = st.columns([1, 1.6], gap="large")

    with col_in3:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                    font-size:12px;letter-spacing:0.1em;text-transform:uppercase;
                    color:#9F9F9F;margin-bottom:16px;">Athlete Details</div>""",
                    unsafe_allow_html=True)
        name3 = st.text_input("Athlete Name", placeholder="e.g. Alex Johnson", key="rs_name")
        col_a3, col_b3 = st.columns(2)
        with col_a3:
            test_date3 = st.date_input("Test Date", key="rs_date")
        with col_b3:
            bw = st.number_input("Body Mass (kg)", 0.0, 300.0, 70.0, step=0.5, key="rs_bw")

        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                    font-size:12px;letter-spacing:0.1em;text-transform:uppercase;
                    color:#9F9F9F;margin:20px 0 12px;">Lift Data</div>""",
                    unsafe_allow_html=True)

        lifts_data = [
            ("Squat",       "rs_sq_r",  "rs_sq_l"),
            ("Bench",       "rs_be_r",  "rs_be_l"),
            ("Deadlift",    "rs_dl_r",  "rs_dl_l"),
            ("Power Clean", "rs_pc_r",  "rs_pc_l"),
            ("Pull-Up",     "rs_pu_r",  "rs_pu_l"),
        ]
        defaults_reps = [5, 5, 5, 5, 5]
        defaults_load = [100.0, 60.0, 140.0, 80.0, 0.0]

        reps_vals = []
        load_vals = []
        for i, (lift, rk, lk) in enumerate(lifts_data):
            st.markdown(f"""<div style="font-size:11px;font-weight:600;letter-spacing:0.05em;
                        text-transform:uppercase;color:#9F9F9F;
                        margin:10px 0 4px;">{lift}</div>""",
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                r = st.number_input("Reps", 0, 20, defaults_reps[i], key=rk)
            with c2:
                label_l = "Add. load (kg)" if lift == "Pull-Up" else "Load (kg)"
                l = st.number_input(label_l, 0.0, 500.0, defaults_load[i], key=lk)
            reps_vals.append(r)
            load_vals.append(l)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_out3:
        params3 = (bw,
                   reps_vals[0], load_vals[0],
                   reps_vals[1], load_vals[1],
                   reps_vals[2], load_vals[2],
                   reps_vals[3], load_vals[3],
                   reps_vals[4], load_vals[4])
        df3 = compute_tab3(params3)
        ratios3  = df3['1RM / BW'].tolist()
        max_val3 = max(ratios3) * 1.25 if any(r > 0 for r in ratios3) else 3.0
        lifts3   = ["Squat","Bench","Deadlift","Power Clean","Pull-Up"]

        fig3 = make_radar(lifts3, ratios3, PERF, max_val=round(max_val3, 1))
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("""<div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                    font-size:12px;letter-spacing:0.1em;text-transform:uppercase;
                    color:#9F9F9F;margin-bottom:8px;">Strength Profile</div>""",
                    unsafe_allow_html=True)
        _cols3 = df3.columns.tolist()
        _rows3 = df3.values.tolist()
        _hdr3 = "".join(f'<th style="padding:10px 14px;font-family:\'Barlow Condensed\',sans-serif;font-weight:700;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;color:#9F9F9F;background:#1A2B4A;border-bottom:1px solid rgba(255,255,255,0.07);">{c}</th>' for c in _cols3)
        _body3 = ""
        for _i, _row in enumerate(_rows3):
            _bg = "rgba(255,255,255,0.025)" if _i % 2 == 0 else "rgba(255,255,255,0.055)"
            _cells = "".join(f'<td style="padding:10px 14px;font-size:13px;color:rgba(255,255,255,0.9);border-top:1px solid rgba(255,255,255,0.07);">{v}</td>' for v in _row)
            _body3 += f'<tr style="background:{_bg}">{_cells}</tr>'
        st.markdown(f'''<table style="width:100%;border-collapse:collapse;border-radius:10px;overflow:hidden;border:1px solid rgba(255,255,255,0.07);">
          <thead><tr>{_hdr3}</tr></thead><tbody>{_body3}</tbody></table>''', unsafe_allow_html=True)

        info3 = {
            "Athlete": name3 or "—",
            "Test Date": test_date3.strftime('%d %b %Y'),
            "Body Mass": f"{bw} kg",
        }
        pdf3 = create_pdf("Relative Strength Report", info3, df3,
                          chart_labels=lifts3, chart_values=ratios3, max_val=round(max_val3,1))
        st.download_button("⬇ Download PDF Report", pdf3,
                           "relative_strength_report.pdf", "application/pdf",
                           key="dl_rs")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px;padding-top:20px;
            border-top:1px solid rgba(255,255,255,0.07);
            display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
  <div style="width:8px;height:8px;border-radius:50%;background:#23FF00;flex-shrink:0;"></div>
  <p style="font-size:11px;color:#4A5568;letter-spacing:0.05em;margin:0;">
    * Kettlebells introduced once technical competence has been demonstrated. &nbsp;·&nbsp;
    Framework guided by LTAD/YPD models. &nbsp;·&nbsp;
    <a href="https://www.ltadnetwork.com" target="_blank"
       style="color:#23FF00;text-decoration:none;">ltadnetwork.com</a>
  </p>
</div>
<div style="text-align:center;margin-top:16px;font-size:11px;color:#4A5568;
            letter-spacing:0.06em;">
  © LTAD Network &nbsp;·&nbsp; Youth Strength Assessment App &nbsp;·&nbsp;
  Strength Development Framework v2.0
</div>
""", unsafe_allow_html=True)
