import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def create_3d_surface_chart(df, x_col, y_col, z_col, color_col):
    """Cria gráfico de superfície 3D"""
    fig = go.Figure(data=[go.Surface(
        z=df.pivot_table(index=y_col, columns=x_col, values=z_col).values,
        x=df[x_col].unique(),
        y=df[y_col].unique(),
        colorscale='Viridis'
    )])

    fig.update_layout(
        title='Gráfico de Superfície 3D',
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        )
    )
    return fig


def create_animated_bubble_chart(df, x_col, y_col, size_col, color_col, animation_col):
    """Cria gráfico de bolhas animado"""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        color=color_col,
        animation_frame=animation_col,
        size_max=50,
        title='Gráfico de Bolhas Animado'
    )
    return fig


def create_sankey_diagram(df, source_col, target_col, value_col):
    """Cria diagrama de Sankey para fluxos"""
    # Criar labels únicas
    labels = list(pd.unique(df[[source_col, target_col]].values.ravel()))

    # Mapear labels para índices
    source_indices = [labels.index(x) for x in df[source_col]]
    target_indices = [labels.index(x) for x in df[target_col]]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="blue"
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=df[value_col]
        )
    )])

    fig.update_layout(title='Diagrama de Sankey - Fluxo de Clientes')
    return fig