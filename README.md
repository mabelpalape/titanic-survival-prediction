# 🚢 Predicción de Supervivencia – Titanic

Aplicación web desarrollada con **Streamlit** para predecir las probabilidades de supervivencia de un pasajero del Titanic utilizando un modelo pre-entrenado guardado con **PyCaret** (`ExtraTreesClassifier`).

---

## 📌 Características del Proyecto

- **Modelo Existente**: Utiliza un modelo pre-entrenado con PyCaret (`modelo/modelo_supervivencia_titanic.pkl`) sin necesidad de volver a entrenar.
- **Interfaz Interactiva**: Construida con Streamlit, ofreciendo un diseño moderno y accesible.
- **Predicción y Probabilidad**: Muestra la probabilidad estimada de supervivencia en porcentaje y un resultado claro (*probablemente sobreviviría* / *probablemente no sobreviviría*).
- **Acceso en Red Local**: Configurada para ejecutarse y recibir conexiones desde cualquier dispositivo (computador, tablet, teléfono) conectado a la misma red Wi-Fi.

---

## 📁 Estructura del Proyecto

```
titanic-survival/
│
├── app.py                  # Aplicación principal de Streamlit
├── modelo/
│   └── modelo_supervivencia_titanic.pkl  # Modelo entrenado con PyCaret
├── data/
│   └── semana-4.dbc        # Archivo fuente original / cuaderno Databricks
├── requirements.txt        # Dependencias necesarias
├── README.md               # Documentación del proyecto
└── .gitignore              # Archivos ignorados por Git
```

---

## 🛠️ Instalación y Requisitos

### 1. Clonar el repositorio
```bash
git clone https://github.com/mabelpalape/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. Crear y activar un entorno virtual
En macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows:
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## 🚀 Ejecución Local

Para iniciar la aplicación en tu máquina local:

```bash
streamlit run app.py
```

Luego abre en tu navegador: [http://localhost:8501](http://localhost:8501)

---

## 🌐 Acceso desde la Red Local

Para permitir que otros computadores o dispositivos móviles conectados a la misma red Wi-Fi accedan a la aplicación, ejecuta:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Abre la dirección con tu IP local desde cualquier dispositivo en la misma red:

```
http://192.168.1.84:8501
```

*(Nota: Reemplaza `192.168.1.84` por la IP local de la máquina si cambia).*

---

## 📊 Variables de Entrada del Modelo

La aplicación solicita las siguientes variables:
- **Clase del pasajero (`Pclass`)**: 1ª, 2ª o 3ª clase.
- **Sexo (`Sex`)**: Femenino / Masculino.
- **Edad (`Age`)**: Edad en años.
- **Hermanos/Esposos a bordo (`SibSp`)**: Cantidad a bordo.
- **Padres/Hijos a bordo (`Parch`)**: Cantidad a bordo.
- **Tarifa (`Fare`)**: Precio del billete.
- **Puerto de embarque (`Embarked`)**: Cherburgo (C), Queenstown (Q) o Southampton (S).
