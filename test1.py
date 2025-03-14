# Importamos las librerías necesarias para el análisis
import pandas as pd  # Manipulación de datos tabulares
import numpy as np  # Operaciones numéricas rápidas
import random  # Generación de valores aleatorios
from faker import Faker  # Creación de datos sintéticos realistas
import os  # Manejo del sistema de archivos
import matplotlib.pyplot as plt  # Visualización básica
import seaborn as sns  # Visualización avanzada y atractiva
from sklearn.model_selection import train_test_split  # División de datos en entrenamiento/prueba
from sklearn.ensemble import RandomForestClassifier  # Modelo de clasificación basado en árboles
from sklearn.metrics import classification_report, confusion_matrix  # Métricas de evaluación del modelo
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Preprocesamiento de variables

# Configuramos Faker para texto en inglés consistente con el análisis de sentimiento posterior
fake = Faker('en_US')  # Especificamos inglés para compatibilidad con TextBlob

# Definimos categorías de quejas relacionadas con áreas laborales
categories = ['finanzas', 'recursos_humanos', 'operaciones']

# Generamos datos sintéticos simulando empleados y sus características
def generate_synthetic_data(num_records=3000):
    # Lista para almacenar registros
    data = []
    for _ in range(num_records):
        # Seleccionamos una categoría de queja al azar
        category = random.choice(categories)
        # Generamos un mensaje ficticio de 3 oraciones como queja del empleado
        message = fake.paragraph(nb_sentences=3)
        # Edad del empleado entre 20 y 60 años
        age = random.randint(20, 60)
        # Años de experiencia limitados por la edad (máximo edad - 18)
        years_of_experience = random.randint(1, min(20, age - 18))
        # Ingreso mensual en dólares, rango típico para empleados
        monthly_income = random.randint(1000, 5000)
        # Distancia recorrida al trabajo en kilómetros (100 a 10,000 km)
        distance_traveled = random.randint(100, 10000)
        # Nivel de riesgo laboral: alta, media o baja
        risk_zone = random.choice(['alta', 'media', 'baja'])
        # Género del empleado
        gender = random.choice(['male', 'female'])
        # Nivel educativo alcanzado
        education_level = random.choice(['high_school', 'bachelor', 'master', 'phd'])
        # Tipo de vehículo usado para commuting
        vehicle_type = random.choice(['sedan', 'suv', 'motorcycle', 'van'])
        # Número de hijos, entre 0 y 5
        number_of_children = random.randint(0, 5)
        # Estado civil del empleado
        marital_status = random.choice(['single', 'married', 'divorced', 'widowed'])
        # Horas promedio trabajadas por semana (20 a 60)
        average_hours_worked_per_week = random.randint(20, 60)
        # Calificación del empleado por clientes, entre 3.0 y 5.0
        customer_rating = round(random.uniform(3.0, 5.0), 1)
        # Viajes realizados el último mes (10 a 100)
        number_of_trips_last_month = random.randint(10, 100)
        # Duración promedio de cada viaje en minutos (10 a 120)
        average_trip_duration = random.randint(10, 120)
        # Región geográfica del empleado
        region = random.choice(['north', 'south', 'east', 'west'])
        
        # Agregamos el registro a la lista como una fila
        data.append([category, message, age, years_of_experience, monthly_income, distance_traveled, 
                     risk_zone, gender, education_level, vehicle_type, number_of_children, 
                     marital_status, average_hours_worked_per_week, customer_rating, 
                     number_of_trips_last_month, average_trip_duration, region])
    
    # Convertimos la lista en un DataFrame con nombres de columnas descriptivos
    return pd.DataFrame(data, columns=['category', 'message', 'age', 'years_of_experience', 
                                       'monthly_income', 'distance_traveled', 'risk_zone', 
                                       'gender', 'education_level', 'vehicle_type', 'number_of_children', 
                                       'marital_status', 'average_hours_worked_per_week', 
                                       'customer_rating', 'number_of_trips_last_month', 
                                       'average_trip_duration', 'region'])

# Generamos los datos sintéticos
df = generate_synthetic_data()

# Creamos el directorio 'data' si no existe para almacenar el archivo
if not os.path.exists('data'):
    os.makedirs('data')

# Guardamos los datos en un CSV para reutilización, con manejo de errores
try:
    df.to_csv('data/synthetic_data.csv', index=False)
    print("Datos guardados exitosamente en 'data/synthetic_data.csv'.")
except Exception as e:
    print(f"Error al guardar el archivo: {e}")
    exit()

# Cargamos los datos desde el CSV para simular un flujo realista
try:
    df = pd.read_csv('data/synthetic_data.csv')
    print("Datos cargados correctamente desde el CSV.")
    print(df.head())  # Mostramos las primeras 5 filas para inspección
except FileNotFoundError:
    print("No se encontró el archivo 'synthetic_data.csv'. Revisa la ruta.")
    exit()

# Realizamos un análisis exploratorio inicial para entender los datos
print("\n--- Análisis Exploratorio de Datos (EDA) ---")
print(f"Dimensiones del dataset: {df.shape}")  # Filas y columnas
print("\nEstadísticas descriptivas de variables numéricas:")
print(df.describe())  # Resumen de media, mediana, etc.
print("\nDistribución de categorías de quejas:")
print(df['category'].value_counts())  # Conteo por categoría

# Visualizamos la distribución de las categorías de quejas
plt.figure(figsize=(8, 5))
sns.countplot(x='category', data=df, palette='viridis')
plt.title("Distribución de Categorías de Quejas", fontsize=14)
plt.xlabel("Categoría", fontsize=12)
plt.ylabel("Cantidad", fontsize=12)
plt.show()

# Calculamos y visualizamos correlaciones solo para variables numéricas
numeric_df = df.select_dtypes(include=['int64', 'float64'])  # Filtramos columnas numéricas
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Matriz de Correlación entre Variables Numéricas", fontsize=14)
plt.show()

# Preprocesamiento: Creamos la variable objetivo antes de normalizar
df['high_turnover'] = ((df['monthly_income'] < 2000) & (df['risk_zone'] == 'alta')).astype(int)
print("\nDistribución de la variable objetivo 'high_turnover' (1: alta rotación, 0: baja):")
print(df['high_turnover'].value_counts())

# Codificamos variables categóricas individualmente para evitar conflictos
encoders = {}  # Diccionario para guardar encoders por columna
for col in ['category', 'risk_zone', 'gender', 'education_level', 'vehicle_type', 'marital_status', 'region']:
    encoders[col] = LabelEncoder()  # Creamos un encoder específico
    df[col] = encoders[col].fit_transform(df[col])  # Transformamos la columna

# Normalizamos las variables numéricas para estandarizar escalas
scaler = StandardScaler()
numerical_features = ['age', 'years_of_experience', 'monthly_income', 'distance_traveled', 
                      'average_hours_worked_per_week', 'customer_rating', 'number_of_trips_last_month', 
                      'average_trip_duration']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Preparamos los datos para el modelo: características (X) y objetivo (y)
X = df.drop(['high_turnover', 'message'], axis=1)  # Excluimos 'message' por ser texto crudo
y = df['high_turnover']

# Dividimos en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos y entrenamos un modelo Random Forest para predecir rotación
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Modelo entrenado con éxito.")

# Evaluamos el modelo con los datos de prueba
y_pred = model.predict(X_test)
print("\n--- Evaluación del Modelo ---")
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))  # Métricas como precisión, recall, F1
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))  # Distribución de predicciones

# Visualizamos la importancia de las características en el modelo
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importances.sort_values(ascending=False).plot(kind='bar', color='teal')
plt.title("Importancia de las Características en la Predicción de Rotación", fontsize=14)
plt.xlabel("Características", fontsize=12)
plt.ylabel("Importancia", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.show()

# Graficamos la matriz de confusión con etiquetas claras
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Baja', 'Alta'], yticklabels=['Baja', 'Alta'])
plt.xlabel("Predicción", fontsize=12)
plt.ylabel("Real", fontsize=12)
plt.title("Matriz de Confusión del Modelo", fontsize=14)
plt.show()

# Exploramos relaciones entre variables clave para storytelling
plt.figure(figsize=(10, 6))
sns.scatterplot(x='distance_traveled', y='monthly_income', hue='high_turnover', 
                data=df, palette='deep', alpha=0.6)
plt.title("Distancia Recorrida vs. Ingreso Mensual por Rotación", fontsize=14)
plt.xlabel("Distancia Recorrida (normalizada)", fontsize=12)
plt.ylabel("Ingreso Mensual (normalizado)", fontsize=12)
plt.legend(title="Rotación", labels=['Baja', 'Alta'])
plt.show()

# Analizamos el sentimiento de los mensajes (solo en inglés por TextBlob)
from textblob import TextBlob  # Importamos aquí para claridad en el flujo
df['message_sentiment'] = df['message'].apply(lambda x: TextBlob(x).sentiment.polarity)
plt.figure(figsize=(10, 6))
sns.histplot(df['message_sentiment'], bins=30, kde=True, color='purple')
plt.title("Distribución del Sentimiento en los Mensajes de Quejas", fontsize=14)
plt.xlabel("Polaridad del Sentimiento (-1: Negativo, 1: Positivo)", fontsize=12)
plt.ylabel("Frecuencia", fontsize=12)
plt.show()