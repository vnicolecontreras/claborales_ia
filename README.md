# Sistema RAG de Consultas Laborales

## Descripción del proyecto

Este proyecto corresponde a una aplicación práctica basada en la arquitectura RAG (Retrieval-Augmented Generation) para responder consultas laborales utilizando como fuente principal el documento:

**Manual 333 Preguntas y Respuestas Laborales | Actualizado Marzo 2026**

La aplicación permite ingresar una consulta en lenguaje natural, buscar fragmentos relevantes dentro del PDF cargado y entregar una respuesta basada únicamente en la evidencia encontrada en el documento.

## Objetivo

Desarrollar una solución de apoyo para consultas laborales que permita:

- recuperar información desde una base documental,
- reducir respuestas inventadas por el modelo,
- mejorar la trazabilidad de la información,
- y entregar respuestas más confiables dentro del contexto laboral chileno.

## Tecnologías utilizadas

- Python
- Streamlit
- LangChain
- PyPDF
- Búsqueda por recuperación de fragmentos en texto

## Estructura del proyecto

```text
chatbot_nuevo_pensiones/
│
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── 1774696815484.pdf
└── src/
    └── rag_engine.py