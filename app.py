# =====================================================================
# BLOQUE 1: IMPORTS
# =====================================================================
import streamlit as st
import pandas as pd
import os
import kagglehub
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_experimental.tools import PythonAstREPLTool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser

# =====================================================================
# BLOQUE 2: CONFIGURACIÓN INICIAL
# =====================================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("⚠️ No se encontró GROQ_API_KEY en el archivo .env")
    st.stop()

st.set_page_config(page_title="Agente Fintech con RAG", layout="wide")
st.title("🏦 Agente de IA para Fintech Mexicana")
st.markdown("---")

# =====================================================================
# BLOQUE 3: GENERACIÓN DE DATOS SINTÉTICOS (sin Kaggle)
# =====================================================================
import numpy as np
from datetime import datetime, timedelta

@st.cache_resource
def generar_datos_sinteticos():
    """
    Genera 4 dataframes típicos de una fintech:
    - clientes: información de clientes
    - cuentas: cuentas asociadas
    - movimientos: transacciones de los últimos 6 meses
    - tarjetas: tarjetas de crédito/débito
    """
    np.random.seed(42)  # Para reproducibilidad
    num_clientes = 50
    num_movimientos = 2000
    
    # --- 1. CLIENTES ---
    nombres = ["Juan Pérez", "María García", "Carlos López", "Ana Martínez", "Luis Rodríguez",
               "Laura Sánchez", "Pedro Gómez", "Marta Fernández", "José Díaz", "Isabel Ruiz",
               "Miguel Ángel", "Carmen Jiménez", "Javier Moreno", "Elena Muñoz", "Antonio Romero",
               "Raquel Alonso", "Manuel Castro", "Sofía Ortega", "David Navarro", "Clara Delgado",
               "Alejandro Ramos", "Nuria Serrano", "Pablo Vázquez", "Cristina Reyes", "Hugo Gutiérrez",
               "Silvia Costa", "Adrián Herrera", "Eva Núñez", "Óscar Luna", "Mónica Fuentes"]
    # Aseguramos 50 clientes (rellenamos con nombres repetidos si hace falta)
    while len(nombres) < num_clientes:
        nombres += nombres[:num_clientes - len(nombres)]
    nombres = nombres[:num_clientes]
    
    clientes = pd.DataFrame({
        "id_cliente": range(1, num_clientes + 1),
        "nombre": nombres,
        "email": [f"cliente{i}@fintech.mx" for i in range(1, num_clientes + 1)],
        "fecha_registro": [datetime.now() - timedelta(days=np.random.randint(1, 730)) for _ in range(num_clientes)],
        "rfc": [f"RFC{i:05d}XYZ" for i in range(1, num_clientes + 1)],
        "telefono": [f"55{np.random.randint(1000,9999)}-{np.random.randint(1000,9999)}" for _ in range(num_clientes)],
        "nivel_ingresos": np.random.choice(["Bajo", "Medio", "Alto"], num_clientes, p=[0.2, 0.6, 0.2]),
        "puntaje_credito": np.random.randint(400, 850, num_clientes)
    })
    
    # --- 2. CUENTAS ---
    # Cada cliente puede tener 1 o 2 cuentas (60% una, 40% dos)
    cuentas = []
    for cid in range(1, num_clientes + 1):
        num_cuentas = 1 if np.random.rand() < 0.6 else 2
        for i in range(num_cuentas):
            cuentas.append({
                "id_cuenta": len(cuentas) + 1,
                "id_cliente": cid,
                "tipo": np.random.choice(["Débito", "Crédito"]),
                "saldo": round(np.random.uniform(1000, 50000), 2),
                "fecha_apertura": datetime.now() - timedelta(days=np.random.randint(1, 730)),
                "estatus": np.random.choice(["Activa", "Inactiva", "Bloqueada"], p=[0.85, 0.1, 0.05])
            })
    cuentas = pd.DataFrame(cuentas)
    
    # --- 3. MOVIMIENTOS (últimos 6 meses) ---
    fechas_inicio = datetime.now() - timedelta(days=180)
    movs = []
    for _ in range(num_movimientos):
        fecha = fechas_inicio + timedelta(days=np.random.randint(0, 181))
        cuenta = cuentas.sample(1).iloc[0]
        monto = round(np.random.uniform(10, 5000), 2)
        if np.random.rand() < 0.3:  # 30% de los movimientos son negativos (retiros/compras)
            monto = -monto
        movs.append({
            "id_movimiento": len(movs) + 1,
            "id_cuenta": cuenta["id_cuenta"],
            "fecha": fecha,
            "monto": monto,
            "tipo": np.random.choice(["Transferencia", "Pago con tarjeta", "Depósito", "Retiro"]),
            "descripcion": np.random.choice(["Compra en tienda", "Pago de servicios", "Transferencia a terceros", "Depósito en efectivo", "Retiro en cajero"]),
            "categoria": np.random.choice(["Alimentación", "Transporte", "Entretenimiento", "Vivienda", "Salud", "Educación"])
        })
    movimientos = pd.DataFrame(movs)
    movimientos["fecha"] = pd.to_datetime(movimientos["fecha"])
    
    # --- 4. TARJETAS (asociadas a cuentas de crédito) ---
    tarjetas = []
    for _, cuenta in cuentas[cuentas["tipo"] == "Crédito"].iterrows():
        tarjetas.append({
            "id_tarjeta": len(tarjetas) + 1,
            "id_cuenta": cuenta["id_cuenta"],
            "numero": f"****-****-****-{np.random.randint(1000,9999)}",
            "limite_credito": round(np.random.uniform(5000, 50000), 2),
            "saldo_actual": round(np.random.uniform(0, 20000), 2),
            "fecha_vencimiento": datetime.now() + timedelta(days=np.random.randint(30, 1095))
        })
    tarjetas = pd.DataFrame(tarjetas)
    
    # Devolver todos los dataframes en un diccionario
    return {
        "clientes": clientes,
        "cuentas": cuentas,
        "movimientos": movimientos,
        "tarjetas": tarjetas
    }

dataframes = generar_datos_sinteticos()

# --- VALIDACIÓN Y SELECCIÓN ---
if not dataframes:
    st.error("❌ No se generaron datos sintéticos.")
    st.stop()

st.sidebar.header("📁 Selecciona el dataset")
claves = list(dataframes.keys())
nombre_seleccionado = st.sidebar.selectbox("Tabla:", claves, index=0)

if nombre_seleccionado is None or nombre_seleccionado not in dataframes:
    nombre_seleccionado = claves[0]

df = dataframes[nombre_seleccionado]
st.sidebar.success(f"✅ {nombre_seleccionado} cargado ({df.shape[0]} filas)")

st.subheader("📊 Vista previa de los datos")
st.dataframe(df.head())
# =====================================================================
# BLOQUE 4: CONFIGURACIÓN DEL RAG (Documentos CONDUSEF)
# =====================================================================
@st.cache_resource
def cargar_documentos_rag():
    """Carga los PDFs de la carpeta 'documentos_rag', los divide en chunks y crea el vectorstore."""
    carpeta = "documentos_rag"
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
        st.warning(f"📁 No se encontró la carpeta '{carpeta}'. Por favor, descarga los 5 PDFs y colócalos ahí.")
        return None
    
    # Cargar todos los PDFs
    documentos = []
    for archivo in os.listdir(carpeta):
        if archivo.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(carpeta, archivo))
            documentos.extend(loader.load())
    
    if not documentos:
        st.warning("⚠️ No se encontraron PDFs en la carpeta 'documentos_rag'.")
        return None
    
    # Dividir en chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documentos)
    
    # Crear embeddings y vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    vectorstore.persist()
    return vectorstore

vectorstore = cargar_documentos_rag()

# =====================================================================
# BLOQUE 5: INSTANCIAR LLM (Groq)
# =====================================================================
llm = ChatGroq(
    api_key=GROQ_API_KEY,

    # model_name="llama3-70b-8192", Se cambio a un modelo más versátil y reciente
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# =====================================================================
# BLOQUE 6: HERRAMIENTAS DEL AGENTE (Datos + RAG)
# =====================================================================
def crear_herramientas(df, vectorstore):
    """Crea las herramientas que usará el agente."""
    
    # ---- Herramienta 1: Información general del DataFrame (adaptada) ----
    def informacion_df(pregunta: str) -> str:
        shape = df.shape
        columns = df.dtypes
        nulos = df.isnull().sum()
        nans_str = df.apply(lambda col: col[~col.isna()].astype(str).str.strip().str.lower().eq('nan').sum())
        duplicados = df.duplicated().sum()
        
        plantilla = PromptTemplate(
            template="""
            Eres un asesor financiero. Analiza estos datos del cliente y da un resumen claro.
            
            Información:
            - Dimensiones: {shape}
            - Columnas: {columns}
            - Datos nulos: {nulos}
            - Filas duplicadas: {duplicados}
            
            Pregunta: {pregunta}
            
            Responde de forma amigable y útil.
            """,
            input_variables=['pregunta','shape','columns','nulos','nans_str','duplicados']
        )
        cadena = plantilla | llm | StrOutputParser()
        return cadena.invoke({
            "pregunta": pregunta,
            "shape": shape,
            "columns": columns,
            "nulos": nulos,
            "nans_str": nans_str,
            "duplicados": duplicados
        })
    
    # ---- Herramienta 2: Estadísticas descriptivas ----
    def resumen_estadistico(pregunta: str) -> str:
        resumen = df.describe(include='number').transpose().to_string()
        plantilla = PromptTemplate(
            template="""
            Eres un asesor financiero. Interpreta estas estadísticas.
            
            Estadísticas:
            {resumen}
            
            Pregunta: {pregunta}
            
            Da una explicación clara y destaca valores importantes.
            """,
            input_variables=['pregunta','resumen']
        )
        cadena = plantilla | llm | StrOutputParser()
        return cadena.invoke({"pregunta": pregunta, "resumen": resumen})
    
    # ---- Herramienta 3: Generar gráficos ----
    def generar_grafico(pregunta: str) -> str:
        columnas_info = '\n'.join([f"- {col} ({dtype})" for col,dtype in df.dtypes.items()])
        muestra = df.head(3).to_dict(orient='records')
        
        plantilla = PromptTemplate(
            template="""
            Eres un experto en visualización de datos financieros.
            Genera **solo el código Python** para el gráfico solicitado.
            
            Solicitud: {pregunta}
            Columnas: {columnas}
            Muestra: {muestra}
            
            Reglas:
            - Usa `plt` y `sns`
            - Elige el tipo adecuado (barplot, lineplot, histplot, etc.)
            - Añade título y etiquetas
            - Termina con `plt.show()`
            
            Devuelve **solo el código Python**.
            """,
            input_variables=['pregunta','columnas','muestra']
        )
        cadena = plantilla | llm | StrOutputParser()
        script_bruto = cadena.invoke({"pregunta": pregunta, "columnas": columnas_info, "muestra": muestra})
        script_limpio = script_bruto.replace("```python", "").replace("```", "").strip()
        
        exec_globals = {"df": df, "plt": plt, "sns": sns}
        exec_locals = {}
        exec(script_limpio, exec_globals, exec_locals)
        fig = plt.gcf()
        st.pyplot(fig)
        return "✅ Gráfico generado."
    
    # ---- Herramienta 4: Búsqueda en documentos (RAG) ----
    def buscar_en_documentos(pregunta: str) -> str:
        if vectorstore is None:
            return "⚠️ La base de documentos no está disponible. Descarga los PDFs en la carpeta 'documentos_rag'."
        docs = vectorstore.similarity_search(pregunta, k=4)
        contexto = "\n\n".join([doc.page_content for doc in docs])
        plantilla = PromptTemplate(
            template="""
            Eres un asesor financiero experto en la regulación mexicana.
            Usa el siguiente contexto para responder la pregunta del cliente.
            Si no encuentras la respuesta, indícalo y sugiere contactar a un ejecutivo.
            
            Contexto:
            {contexto}
            
            Pregunta: {pregunta}
            
            Respuesta (en español, clara y profesional):
            """,
            input_variables=['contexto','pregunta']
        )
        cadena = plantilla | llm | StrOutputParser()
        return cadena.invoke({"contexto": contexto, "pregunta": pregunta})
    
    # ---- Herramienta 5: Código Python para cálculos puntuales ----
    herramienta_codigo = Tool(
        name="Python REPL",
        func=PythonAstREPLTool(locals={"df": df}),
        description="Úsala para cálculos exactos (promedios, filtros, sumas) sobre el DataFrame `df`.",
        return_direct=False
    )
    
    # ---- Empaquetar todas las herramientas ----
    herramientas = [
        Tool(name="Información General", func=informacion_df, description="Resumen general del dataset", return_direct=True),
        Tool(name="Estadísticas Descriptivas", func=resumen_estadistico, description="Estadísticas de columnas numéricas", return_direct=True),
        Tool(name="Generar Gráfico", func=generar_grafico, description="Crea gráficos a partir de preguntas", return_direct=True),
        Tool(name="Buscar en Documentos", func=buscar_en_documentos, description="Busca en leyes y reglamentos de la CONDUSEF", return_direct=True),
        herramienta_codigo
    ]
    return herramientas

tools = crear_herramientas(df, vectorstore)

# =====================================================================
# BLOQUE 7: CREACIÓN DEL AGENTE REACT
# =====================================================================
prompt_react = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    partial_variables={"df_head": df.head().to_markdown()},
    template="""
    Eres un asesor financiero experto en el sector bancario mexicano.
    Hablas en español y siempre das respuestas claras, precisas y amables.
    
    Tienes acceso a un DataFrame `df` con los datos del cliente (movimientos, saldos, etc.).
    Estas son las primeras filas:
    {df_head}
    
    También tienes acceso a documentos legales de la CONDUSEF a través de la herramienta "Buscar en Documentos".
    
    **INSTRUCCIONES:**
    - Si el cliente pregunta sobre leyes, comisiones, derechos o procedimientos, usa "Buscar en Documentos".
    - Si pregunta sobre sus datos (saldos, gastos, promedios), usa "Información General", "Estadísticas Descriptivas" o "Python REPL" según corresponda.
    - Si pide un gráfico, usa "Generar Gráfico".
    - Si no encuentras la información, dilo claramente y sugiere escalar a un ejecutivo.
    
    **HERRAMIENTAS DISPONIBLES:**
    {tools}
    
    **FORMATO DE RESPUESTA:**
    Question: la pregunta del cliente
    Thought: piensa qué herramienta usar
    Action: nombre de la herramienta (debe ser una de [{tool_names}])
    Action Input: entrada para la herramienta
    Observation: resultado de la herramienta
    ... (puedes repetir si es necesario)
    Thought: ahora sé la respuesta final
    Final Answer: respuesta final para el cliente
    
    Comienza:
    Question: {input}
    Thought: {agent_scratchpad}
    """
)

agente = create_react_agent(llm=llm, tools=tools, prompt=prompt_react)
orquestador = AgentExecutor(
    agent=agente,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# =====================================================================
# BLOQUE 8: INTERFAZ DE USUARIO (Streamlit)
# =====================================================================
st.markdown("---")
st.subheader("💬 Haz una pregunta")

# Historial de mensajes
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])

# Input del usuario
if pregunta := st.chat_input("Escribe tu consulta..."):
    # Mostrar mensaje del usuario
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)
    
    # Generar respuesta del agente
    with st.chat_message("assistant"):
        with st.spinner("🦜 Pensando..."):
            try:
                respuesta = orquestador.invoke({"input": pregunta})
                output = respuesta["output"]
            except Exception as e:
                output = f"❌ Ocurrió un error: {str(e)}. Por favor, intenta de nuevo."
            st.markdown(output)
            st.session_state.mensajes.append({"rol": "assistant", "contenido": output})

# =====================================================================
# BLOQUE 9: BOTONES DE ACCIÓN RÁPIDA (en sidebar o expander)
# =====================================================================
with st.sidebar:
    st.markdown("---")
    st.subheader("⚡ Acciones Rápidas")
    
    if st.button("📄 Resumen general de datos"):
        with st.spinner("Generando..."):
            resp = orquestador.invoke({"input": "Dame un resumen general de los datos"})
            st.sidebar.info(resp["output"])
    
    if st.button("📊 Estadísticas descriptivas"):
        with st.spinner("Generando..."):
            resp = orquestador.invoke({"input": "Dame estadísticas descriptivas de las columnas numéricas"})
            st.sidebar.info(resp["output"])
    
    if st.button("📜 Consultar Ley Fintech"):
        with st.spinner("Buscando..."):
            resp = orquestador.invoke({"input": "¿Qué dice la Ley Fintech sobre la protección de datos?"})
            st.sidebar.info(resp["output"])
    
    if st.button("🗑️ Borrar historial"):
        st.session_state.mensajes = []
        st.rerun()
        