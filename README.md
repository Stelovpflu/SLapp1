
# 💳 Fraud Detection App

Aplicación interactiva construida con **Streamlit** para la detección de transacciones
fraudulentas usando un modelo de **Machine Learning (Random Forest Classifier)**,
diseñada para manejar **clases desbalanceadas**, un problema crítico en fraude.

---

## 🚀 Demo
👉 Streamlit Cloud App  
https://sl-app1.streamlit.app/

---

## 🎯 Objetivo del proyecto
Permitir a analistas o equipos de negocio:

- Evaluar transacciones individuales
- Ajustar el **decision threshold**
- Visualizar métricas clave orientadas a fraude
- Entender el trade-off entre **recall y precision**
- Simular escenarios de riesgo

---

## 🧠 Modelo
- Algoritmo: **Random Forest Classifier**
- Enfoque principal:
  - Maximizar **Recall** (detección de fraude)
  - Controlar falsos positivos vía threshold
- Técnicas aplicadas:
  - Manejo de clases desbalanceadas
  - Probabilidades en lugar de predicción binaria fija

### Métricas (test set)
- Recall ≈ **0.85**
- Precision ≈ **0.67**
- Threshold configurable en la app

---

## 📊 Funcionalidades de la App
- Input manual de variables de transacción
- Predicción en tiempo real
- Score de probabilidad de fraude
- Visualización clara del modelo y métricas
- Interfaz pensada para **usuarios no técnicos**

---

## 🛠️ Tecnologías
- Python
- Pandas / NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 📈 Business Impact

La detección de fraude no es un problema de accuracy, sino de **gestión de riesgo**.

Este proyecto está diseñado para apoyar decisiones reales de negocio:

- Reducir pérdidas financieras causadas por transacciones fraudulentas
- Priorizar la detección temprana de fraude (alto recall)
- Permitir ajustar el nivel de tolerancia al riesgo según el contexto del negocio

### Impacto esperado
- Menos fraudes no detectados (falsos negativos)
- Mejor control de falsos positivos mediante threshold
- Flexibilidad para distintos escenarios:
  - Bancos
  - Fintech
  - E-commerce
  - Pagos digitales

La aplicación permite simular cómo cambia el riesgo operativo al modificar el umbral de decisión,
acercando el modelo a la realidad del negocio.

---

## 🧪 Decision Threshold

En problemas de fraude, la probabilidad de fraude es más importante que la predicción binaria.

Por defecto, muchos modelos usan un threshold fijo de 0.5, lo cual **no es óptimo** en escenarios
con clases desbalanceadas.

### Decisión tomada en este proyecto
- El modelo genera probabilidades de fraude
- El objetivo principal es:
  - Maximizar **Recall** (detectar la mayor cantidad de fraude posible)
  - Controlar el impacto en **Precision**

### Ejemplo
- Threshold bajo → más fraude detectado, más falsos positivos
- Threshold alto → menos falsos positivos, mayor riesgo de fraude no detectado

Esta decisión refleja un enfoque **orientado a negocio y riesgo**, no solo a métricas técnicas.

---

## ⚠️ Notas
Este repositorio contiene **únicamente la lógica de inferencia**.
El entrenamiento del modelo y el análisis exploratorio se realizaron por separado.

El enfoque del proyecto está orientado a **casos reales de negocio**, donde:
- El costo de un falso negativo es alto
- El threshold es una decisión estratégica, no técnica

---

## 👤 Autor
**Steve Loveday**  
Data Scientist | Business Analytics | Machine Learning




