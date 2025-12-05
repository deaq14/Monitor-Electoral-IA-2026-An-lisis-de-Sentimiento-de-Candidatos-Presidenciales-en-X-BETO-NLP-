# Monitor-Electoral-IA-2026-An-lisis-de-Sentimiento-de-Candidatos-Presidenciales-en-X-BETO-NLP

Monitor Electoral IA 2026: Análisis de Sentimiento en X (Python, BETO, Dash)
Este proyecto es una herramienta de Ciencia de Datos y Análisis de Redes Sociales diseñada para monitorear, extraer y clasificar la opinión pública en la plataforma X (anteriormente Twitter) sobre figuras políticas clave en el período pre-electoral en Colombia.

El proyecto demuestra el ciclo completo de la Ciencia de Datos: desde la ingesta de datos inestructurados hasta la visualización interactiva.

Objetivo
Cuantificar y visualizar el Sentimiento Neto (balance entre opiniones a favor y en contra) de tres pre-candidatos presidenciales a partir de los comentarios públicos en X durante un período de tiempo específico, proporcionando insights sobre la tendencia de la conversación.

Candidatos Analizados
Abelardo de la Espriella

Iván Cepeda

Sergio Fajardo

Período de Análisis
15 de noviembre de 2025 al 5 de diciembre de 2025.

Stack Tecnológico
Componente	Herramienta	Función
Recolección (ETL)	Python, snscrape (o simulado)	Extracción de datos de la plataforma X.
Limpieza (NLP)	NLTK (Tokenización, Stop Words)	Pre-procesamiento del texto en español.
Clasificación (AI)	Transformers Pipeline, BETO	Modelo de Deep Learning para clasificar el sentimiento (Positivo, Negativo, Neutro).
Visualización	Plotly, Dash	Creación de un Dashboard interactivo para series de tiempo y comparación de Net Sentiment Score.
Ingeniería de Software	logging, Módulos (etl_pipeline.py, dashboard.py)	Manejo de errores robusto y separación de lógica (Backend/Frontend).

Exportar a Hojas de cálculo

Estructura del Proyecto
El código está separado en dos módulos para optimizar la mantenibilidad y la depuración:

etl_pipeline.py (Backend):

Maneja la ingesta de datos (o genera datos simulados si el scraping falla).

Implementa el manejo de errores y el logging centralizado.

Limpia los datos y aplica el modelo BETO.

dashboard.py (Frontend):

Carga los datos procesados desde el pipeline.

Define la estructura del dashboard y los gráficos interactivos de Plotly/Dash.

Ejecución (Modo Simulado)
Debido a las restricciones de la API y los cambios en X, el proyecto está configurado para ejecutarse por defecto en Modo Simulado, utilizando su propio generador de datos si el scraping real falla.

Instalar dependencias:

Bash

pip install pandas nltk transformers torch plotly dash pysentimiento

Ejecutar el dashboard:

Bash

python dashboard.py
Abrir la URL en el navegador: http://127.0.0.1:8050/
