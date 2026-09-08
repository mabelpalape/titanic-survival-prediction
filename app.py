import sys
import types
import os
import pandas as pd
import streamlit as st

# ==============================================================================
# Parche de Compatibilidad (Pandas >= 2.0 y Python 3.12+ con PyCaret)
# ==============================================================================
if 'pandas.core.indexes.numeric' not in sys.modules:
    mod = types.ModuleType('pandas.core.indexes.numeric')
    mod.Int64Index = pd.Index
    sys.modules['pandas.core.indexes.numeric'] = mod

orig_version = sys.version_info
if sys.version_info >= (3, 12):
    sys.version_info = (3, 11, 0, 'final', 0)

import pycaret
from pycaret.classification import load_model, predict_model

sys.version_info = orig_version


# ==============================================================================
# Configuración de la página
# ==============================================================================
st.set_page_config(
    page_title="Predicción de Supervivencia – Titanic",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    /* Estilos generales */
    .main-title {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 800;
        color: #f8fafc;
        text-align: center;
        margin-bottom: 0.2rem;
        font-size: 2.3rem;
    }
    .sub-title {
        color: #94a3b8;
        text-align: center;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
    }
    .result-card-survived {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.25));
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        color: #ecfdf5;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.3);
    }
    .result-card-perished {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.25));
        border: 1px solid rgba(248, 113, 113, 0.4);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        color: #fef2f2;
        box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.3);
    }
    .result-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .probability-badge {
        display: inline-block;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 12px 0;
        padding: 6px 20px;
        border-radius: 12px;
        background: rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.2rem;
        font-size: 1.1rem;
        font-weight: 600;
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39);
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(37, 99, 235, 0.5);
    }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# Cargar el modelo pre-entrenado
# ==============================================================================
@st.cache_resource
def get_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'modelo', 'modelo_supervivencia_titanic')
    return load_model(model_path)

try:
    model = get_model()
except Exception as e:
    st.error(f"Error al cargar el modelo pre-entrenado: {e}")
    st.stop()


# ==============================================================================
# Cabecera principal
# ==============================================================================
st.markdown('<h1 class="main-title">🚢 Predicción de Supervivencia – Titanic</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Ingresa las características del pasajero para estimar sus probabilidades de supervivencia</p>', unsafe_allow_html=True)

st.markdown("---")

# ==============================================================================
# Formulario de entrada
# ==============================================================================
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        pclass_option = st.selectbox(
            "Clase del pasajero (Pclass)",
            options=[1, 2, 3],
            format_func=lambda x: f"{x}ª Clase ({'Primera' if x==1 else 'Segunda' if x==2 else 'Tercera'})",
            help="Clase del billete del pasajero"
        )
        
        sex_option = st.selectbox(
            "Sexo",
            options=["female", "male"],
            format_func=lambda x: "Femenino" if x == "female" else "Masculino"
        )
        
        age = st.number_input(
            "Edad (años)",
            min_value=0.0,
            max_value=100.0,
            value=28.0,
            step=1.0,
            help="Edad del pasajero en años"
        )
        
        embarked_option = st.selectbox(
            "Puerto de embarque (Embarked)",
            options=["C", "Q", "S"],
            format_func=lambda x: "Cherburgo (C)" if x == "C" else "Queenstown (Q)" if x == "Q" else "Southampton (S)"
        )

    with col2:
        sibsp = st.number_input(
            "Hermanos / Esposos a bordo (SibSp)",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Número de hermanos o cónyuges a bordo"
        )
        
        parch = st.number_input(
            "Padres / Hijos a bordo (Parch)",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Número de padres o hijos a bordo"
        )
        
        fare = st.number_input(
            "Tarifa del billete ($)",
            min_value=0.0,
            max_value=600.0,
            value=32.2,
            step=1.0,
            help="Precio abonado por el billete"
        )

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# Botón y lógica de Predicción
# ==============================================================================
if st.button("Predecir supervivencia"):
    # Crear DataFrame de entrada con las características seleccionadas y valores por defecto para metadatos
    input_df = pd.DataFrame([{
        'PassengerId': 1,
        'Pclass': int(pclass_option),
        'Name': 'Pasajero, Pasajero',
        'Sex': sex_option,
        'Age': float(age),
        'SibSp': int(sibsp),
        'Parch': int(parch),
        'Ticket': '1234',
        'Fare': float(fare),
        'Embarked': embarked_option
    }])

    try:
        predictions = predict_model(model, data=input_df)
        
        label = int(predictions['prediction_label'].iloc[0])
        score = float(predictions['prediction_score'].iloc[0])
        percentage = round(score * 100, 1)

        # Ajuste de probabilidad de supervivencia si la clase predicha es 0
        if label == 0:
            survival_prob = round((1.0 - score) * 100, 1)
        else:
            survival_prob = percentage

        st.markdown("<br>", unsafe_allow_html=True)

        if label == 1:
            st.markdown(f"""
                <div class="result-card-survived">
                    <div class="result-title">🟢 Resultado: probablemente sobreviviría</div>
                    <div>Probabilidad estimada de supervivencia</div>
                    <div class="probability-badge">{survival_prob}%</div>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
                <div class="result-card-perished">
                    <div class="result-title">🔴 Resultado: probablemente no sobreviviría</div>
                    <div>Probabilidad estimada de supervivencia</div>
                    <div class="probability-badge">{survival_prob}%</div>
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Ocurrió un error al realizar la predicción: {e}")

# Pie de página
st.markdown("<br><hr><p style='text-align:center; color:#64748b; font-size:0.85rem;'>Modelo entrenado con PyCaret • Titanic Survival Dataset</p>", unsafe_allow_html=True)
