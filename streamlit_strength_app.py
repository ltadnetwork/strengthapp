import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Configure page
st.set_page_config(page_title="Athlete Strength Technical Assessment", layout="wide")
st.title("Athlete Strength Technical Assessment")

# Thresholds for bodyweight test scoring
thresholds = {
    'Full Squat': {
        'M': {'5':40,'4':33,'3':26,'2':19,'1':11,'0':0},
        'F': {'5':40,'4':33,'3':26,'2':19,'1':11,'0':0},
    },
    'Pull-Up': {
        'M': {'5':12,'4':9,'3':6,'2':4,'1':1,'0':0},
        'F': {'5':6,'4':4,'3':3,'2':2,'1':1,'0':0},
    },
    'Push-Up': {
        'M': {'5':35,'4':28,'3':20,'2':10,'1':3,'0':0},
        'F': {'5':20,'4':15,'3':10,'2':5,'1':1,'0':0},
    },
    'Single-Leg Squat Left': {
        'M': {'5':5,'4':4,'3':3,'2':2,'1':1,'0':0},
        'F': {'5':5,'4':4,'3':3,'2':2,'1':1,'0':0},
    },
    'Single-Leg Squat Right': {
        'M': {'5':5,'4':4,'3':3,'2':2,'1':1,'0':0},
        'F': {'5':5,'4':4,'3':3,'2':2,'1':1,'0':0},
    },
    'Front Plank': {
        'M': {'5':121,'4':90,'3':60,'2':30,'1':10,'0':0},
        'F': {'5':121,'4':90,'3':60,'2':30,'1':10,'0':0},
    },
    'Twisting Sit-Up': {
        'M': {'5':40,'4':33,'3':26,'2':19,'1':11,'0':0},
        'F': {'5':40,'4':33,'3':26,'2':19,'1':11,'0':0},
    },
}

def score_value(exercise, val, sex):
    th = thresholds[exercise][sex]
    for p in range(5, -1, -1):
        if val >= th[str(p)]:
            return p
    return 0

# Create tabs
tabs = st.tabs(["Strength Technical Assessment", "Bodyweight Strength Test", "Relative Strength Assessment"])

with tabs[0]:
    # Tab 1 inputs
    col1, col2 = st.columns([1,2])
    with col1:
        athlete_name = st.text_input("Athlete Name")
        athlete_age = st.number_input("Age (years)", min_value=0, value=18)
        predicted_height = st.number_input("% Predicted Adult Height", min_value=0, value=100)
        squat_tech = st.slider("Bodyweight Squat Technique (1-5)", min_value=1, max_value=5, value=3)
        squat_reps = st.number_input("Bodyweight Squat Reps", min_value=0, value=0)
        pushup_tech = st.slider("Push-Up Technique (1-5)", 1,5,3)
        pushup_reps = st.number_input("Push-Up Reps",0,500,0)
        lunge_tech = st.slider("Bodyweight Lunge Technique (1-5)",1,5,3)
        lunge_reps = st.number_input("Bodyweight Lunge Reps",0,500,0)
        row_tech = st.slider("Inverted Row Technique (1-5)",1,5,3)
        row_reps = st.number_input("Inverted Row Reps",0,500,0)
        plank_tech = st.slider("Plank Technique (1-5)",1,5,3)
        plank_time = st.number_input("Plank Hold Time (secs)",0,1000,0)
        sideplank_tech = st.slider("Side Plank Technique (1-5)",1,5,3)
        sideplank_time = st.number_input("Side Plank Hold Time per Side (secs)",0,1000,0)
    with col2:
        st.subheader(f"Athlete: {athlete_name}")
        st.write(f"Age: {athlete_age} years")
        st.write(f"% Predicted Adult Height: {predicted_height}%")
        # DataFrame
        df_tech = pd.DataFrame({
            'Exercise': ["Bodyweight Squat","Push-Up","Bodyweight Lunge","Inverted Row","Plank","Side Plank"],
            'Technique': [squat_tech, pushup_tech, lunge_tech, row_tech, plank_tech, sideplank_tech],
            'Reps/Hold': [squat_reps, pushup_reps, lunge_reps, row_reps, plank_time, sideplank_time],
        })
        # Radar chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=df_tech['Technique'].tolist()+[df_tech['Technique'].iloc[0]],
                                       theta=df_tech['Exercise'].tolist()+[df_tech['Exercise'].iloc[0]],
                                       fill='toself', mode='lines+markers'))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0,5])), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.table(df_tech)

with tabs[1]:
    col1, col2 = st.columns([1,2])
    with col1:
        sex = st.selectbox("Sex", options=["M","F"], format_func=lambda x: "Male" if x=='M' else "Female")
        bw_squat = st.number_input("Full Squat reps in 60 secs (10% BW)",0,500,0)
        pullup = st.number_input("Pull-Up (underhand) reps",0,500,0)
        bw_pushup = st.number_input("Push-Up reps",0,500,0)
        single_left = st.number_input("Single-Leg Squat reps (left)",0,100,0)
        single_right= st.number_input("Single-Leg Squat reps (right)",0,100,0)
        plank_dur = st.number_input("Front Plank duration (secs)",0,1000,0)
        twists = st.number_input("Twisting Sit-Up reps in 60 secs",0,500,0)
    with col2:
        df_bw = pd.DataFrame({
            'Exercise': ["Full Squat","Pull-Up","Push-Up","Single-Leg Squat Left","Single-Leg Squat Right","Front Plank","Twisting Sit-Up"],
            'Value': [bw_squat, pullup, bw_pushup, single_left, single_right, plank_dur, twists]
        })
        df_bw['Score'] = df_bw.apply(lambda row: score_value(row['Exercise'], row['Value'], sex), axis=1)
        # Radar
        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(r=df_bw['Score'].tolist()+[df_bw['Score'].iloc[0]],
                                        theta=df_bw['Exercise'].tolist()+[df_bw['Exercise'].iloc[0]],
                                        fill='toself', mode='lines+markers'))
        fig2.update_layout(polar=dict(radialaxis=dict(range=[0,5])), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.table(df_bw)
        # Total score
        avg_leg = (df_bw.loc[df_bw['Exercise']=="Single-Leg Squat Left", 'Score'].iloc[0] + 
                   df_bw.loc[df_bw['Exercise']=="Single-Leg Squat Right", 'Score'].iloc[0]) / 2
        total = df_bw.loc[~df_bw['Exercise'].isin(["Single-Leg Squat Left","Single-Leg Squat Right"]), 'Score'].sum() + avg_leg
        result = "Pass" if total>18 else "Fail"
        st.table(pd.DataFrame({"Total Score":[round(total,1)], "Result":[result]}))

with tabs[2]:
    st.write("Inputs and outputs for relative strength assessment to be added.")