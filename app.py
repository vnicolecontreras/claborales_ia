from pathlib import Path

import streamlit as st

from src.rag_engine import RagEngine


DEFAULT_PDF = Path("data/1774696815484.pdf")


st.set_page_config(
    page_title="Consultas Laborales RAG",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource
def load_engine(pdf_path: str) -> RagEngine:
    return RagEngine(pdf_path)


st.title("Sistema RAG de Consultas Laborales")

st.write(
    "Esta app busca respuestas dentro del manual laboral y luego arma una respuesta "
    "usando solo ese contenido."
)

st.warning(
    "Uso academico. Esta herramienta responde solo con informacion recuperada del PDF cargado. "
    "Si no encuentra evidencia suficiente, la respuesta no debe considerarse concluyente. "
    "No reemplaza asesoria legal profesional."
)

with st.sidebar:
    st.header("Configuracion")
    pdf_path = st.text_input("Ruta del PDF", value=str(DEFAULT_PDF))
    top_k = st.slider("Cantidad de fragmentos a revisar", min_value=1, max_value=5, value=3)
    st.caption(
        "Si el PDF cambia, presiona el boton para recargar la base del conocimiento."
    )
    reload_clicked = st.button("Recargar documento")

if reload_clicked:
    load_engine.clear()

try:
    engine = load_engine(pdf_path)
except Exception as exc:
    st.error(f"No pude leer el PDF. Revisa la ruta. Detalle: {exc}")
    st.stop()

st.success(f"Documento cargado: {engine.document_name}")
st.caption(f"Fragmentos creados: {len(engine.chunks)}")

example_questions = [
    "Cual es la jornada maxima ordinaria vigente al 26 de abril de 2026?",
    "El empleador esta obligado a escriturar el contrato de trabajo?",
    "Que pasa si el empleador no acompana el estado de pago de cotizaciones en la carta de despido?",
]

selected_example = st.selectbox("Pregunta de ejemplo", [""] + example_questions)
user_question = st.text_area(
    "Escribe tu consulta laboral",
    value=selected_example,
    height=120,
    placeholder="Ejemplo: Que sistemas de control de asistencia son validos?",
)

if st.button("Buscar respuesta", type="primary"):
    if not user_question.strip():
        st.warning("Escribe una pregunta primero.")
        st.stop()

    with st.spinner("Buscando en el manual..."):
        result = engine.ask(user_question, top_k=top_k)

        st.subheader("Respuesta final")
    st.success(result["answer"])

    st.subheader("Trazabilidad principal")
    st.info(f"Fuente principal usada: {result['top_source']}")

    st.subheader("Fuente legal detectada")
    st.caption(result["legal_source"])

    st.subheader("Fragmentos recuperados")
    for index, item in enumerate(result["sources"], start=1):
        with st.expander(f"Fragmento {index}: {item['title']}"):
            st.write(item["text"])
            st.caption(f"Puntaje de coincidencia: {item['score']}")

    st.subheader("Flujo RAG aplicado")
    st.markdown(
        """
        1. **Consulta del usuario**: se recibe la pregunta.
        2. **Recuperacion**: el sistema busca los fragmentos mas relacionados dentro del PDF.
        3. **Contexto**: se seleccionan los fragmentos mas utiles para responder.
        4. **Respuesta**: se entrega una respuesta basada solo en la evidencia encontrada.
        """
    )

    with st.expander("Ver prompt usado por el sistema"):
        st.code(result["prompt"], language="text")

    if result["mode"] != "llm":
        st.info(
            "La respuesta fue generada en modo simple porque no se detectaron "
            "credenciales de modelo. La app igual funciona para mostrar el flujo RAG."
        )