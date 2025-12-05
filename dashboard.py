from dash import Dash, dcc, html
import plotly.express as px
import plotly.graph_objects as go
from etl_pipeline import ejecutar_pipeline # Importamos nuestra lógica separada

# ==========================================
# 1. CARGA DE DATOS DESDE EL PIPELINE
# ==========================================
print("⏳ Ejecutando pipeline de datos... Por favor espere.")
df = ejecutar_pipeline()

if df.empty:
    print("❌ Error: El DataFrame está vacío. Revisa los logs.")
    exit()

# ==========================================
# 2. PREPARACIÓN DE GRÁFICOS (PLOTLY)
# ==========================================

# A. Agrupación para Barras (Score Neto)
summary = df.groupby('Candidate')['Sentiment_Value'].sum().reset_index()

# B. Agrupación para Línea de Tiempo
time_series = df.groupby(['Date_Only', 'Candidate'])['Sentiment_Value'].sum().reset_index()

# Definición de Figuras
def get_fig_bar():
    return px.bar(
        summary, x='Candidate', y='Sentiment_Value',
        color='Sentiment_Value', color_continuous_scale='RdBu',
        title='Balance Neto de Sentimiento (A Favor vs En Contra)',
        template='plotly_dark'
    )

def get_fig_line():
    return px.line(
        time_series, x='Date_Only', y='Sentiment_Value', color='Candidate',
        markers=True, title='Tendencia Temporal de Opinión',
        template='plotly_dark'
    )

def get_fig_pie(candidato):
    data = df[df['Candidate'] == candidato]
    # Contar ocurrencias
    counts = data['Sentiment_Label'].value_counts().reset_index()
    counts.columns = ['Label', 'Count']
    
    return px.pie(
        counts, values='Count', names='Label',
        title=f'{candidato}',
        color='Label',
        color_discrete_map={'POS':'#00CC96', 'NEG':'#EF553B', 'NEU':'#AB63FA'},
        template='plotly_dark'
    )

# ==========================================
# 3. LAYOUT DE LA APLICACIÓN (DASH)
# ==========================================

app = Dash(__name__)
server = app.server # Necesario si despliegas en producción (ej. Heroku/Render)

app.layout = html.Div(style={'backgroundColor': '#111111', 'color': 'white', 'fontFamily': 'Arial', 'padding': '20px'}, children=[
    
    # Encabezado
    html.H1("🗳️ Monitor Electoral 2026", style={'textAlign': 'center'}),
    html.P("Análisis de Sentimiento IA (BETO) sobre Pre-candidatos en X", style={'textAlign': 'center', 'opacity': '0.7'}),
    
    html.Hr(style={'borderColor': '#444'}),

    # Fila Superior: Gráficos Principales
    html.Div([
        html.Div([dcc.Graph(figure=get_fig_bar())], style={'width': '49%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=get_fig_line())], style={'width': '49%', 'display': 'inline-block'}),
    ]),

    # Fila Inferior: Desglose por Candidato
    html.H3("Desglose de Opinión por Candidato", style={'textAlign': 'center', 'marginTop': '30px'}),
    
    html.Div([
        html.Div([dcc.Graph(figure=get_fig_pie("Abelardo de la Espriella"))], style={'width': '33%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=get_fig_pie("Iván Cepeda"))], style={'width': '33%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=get_fig_pie("Sergio Fajardo"))], style={'width': '33%', 'display': 'inline-block'}),
    ]),
    
    # Pie de página
    html.Div([
        html.P(f"Datos procesados: {len(df)} tweets analizados.", style={'marginTop': '20px', 'fontSize': '12px'})
    ])
])

if __name__ == '__main__':
    # debug=True permite recarga automática si cambias el código
    app.run(debug=True, port=8050)