import streamlit as st
import time
import random


def animated_metric(label, value, delta=None):
    """Métrica com animação de contagem"""
    placeholder = st.empty()
    for i in range(0, int(value) + 1, max(1, int(value) // 50)):
        placeholder.metric(label, i, delta)
        time.sleep(0.01)
    placeholder.metric(label, value, delta)


def loading_animation(message="Processando..."):
    """Animação de loading personalizada"""
    with st.spinner(message):
        time.sleep(2)


def success_animation():
    """Animação de sucesso"""
    st.balloons()
    st.snow()
    time.sleep(1)
    st.success("✅ Operação concluída!")


def progress_bar_with_text(iterable, text="Progresso"):
    """Barra de progresso com texto"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, item in enumerate(iterable):
        progress = (i + 1) / len(iterable)
        progress_bar.progress(progress)
        status_text.text(f"{text}: {progress:.1%}")
        time.sleep(0.1)

    progress_bar.empty()
    status_text.empty()