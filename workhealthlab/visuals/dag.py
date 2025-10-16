"""
dag.py - Sociopath-it Causal DAG Visualizations
------------------------------------------------
Directed Acyclic Graphs for causal inference.

Features:
- DAG visualization with networkx
- Confounding path highlighting
- Multiple layout algorithms
- Interactive Plotly version

Note: Requires networkx
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional, List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

try:
    import networkx as nx
except ImportError:
    nx = None
    warnings.warn("networkx not installed. Install with: pip install networkx")

try:
    from ..utils.style import (
        set_style,
        generate_semantic_palette,
        apply_titles,
        COLORS_DICT,
        get_color,
    )
except ImportError:
    def set_style(*args, **kwargs):
        pass
    def apply_titles(*args, **kwargs):
        pass
    def get_color(*args, **kwargs):
        return '#333333'
    COLORS_DICT = {'viridis': plt.cm.viridis}


# ==============================================================================
# DAG
# ==============================================================================

def dag(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    layout: str = "hierarchical",
    highlight_nodes: Optional[List[str]] = None,
    highlight_edges: Optional[List[Tuple[str, str]]] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "viridis",
    figsize: tuple = (10, 8),
    output_path: Optional[str] = None,
):
    """
    Create a directed acyclic graph (DAG) for causal inference.

    Parameters
    ----------
    nodes : list of str
        List of node names (variables).
    edges : list of tuples
        List of directed edges as (source, target) tuples.
    layout : str, default "hierarchical"
        Layout algorithm: 'hierarchical', 'circular', 'spring', 'shell'.
    highlight_nodes : list of str, optional
        Nodes to highlight (e.g., treatment and outcome).
    highlight_edges : list of tuples, optional
        Edges to highlight (e.g., confounding paths).
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "viridis"
        Color scheme.
    figsize : tuple, default (10, 8)
        Figure size.
    output_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> nodes = ['Treatment', 'Outcome', 'Confounder', 'Mediator']
    >>> edges = [('Treatment', 'Outcome'), ('Treatment', 'Mediator'),
    ...          ('Mediator', 'Outcome'), ('Confounder', 'Treatment'),
    ...          ('Confounder', 'Outcome')]
    >>> dag(nodes, edges, highlight_nodes=['Treatment', 'Outcome'],
    ...     title='Causal DAG')
    """
    if nx is None:
        raise ImportError("networkx is required for DAG visualizations. Install with: pip install networkx")

    set_style(style_mode)

    # Create graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Choose layout with improved hierarchical positioning
    if layout == "hierarchical":
        # Try graphviz dot layout first (best for DAGs)
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR')
            # Normalize positions for better display
            if pos:
                x_coords = [p[0] for p in pos.values()]
                y_coords = [p[1] for p in pos.values()]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Normalize to reasonable range
                for node in pos:
                    x, y = pos[node]
                    pos[node] = (
                        (x - x_min) / (x_max - x_min + 0.001) * 3,
                        (y - y_min) / (y_max - y_min + 0.001) * 2
                    )
                print("Using graphviz dot layout for optimal DAG visualization")
        except:
            # Fallback to custom hierarchical layout
            try:
                # Enhanced column-based layout with topological ordering
                # Identify node types by name
                treatment_nodes = [n for n in nodes if 'treatment' in n.lower()]
                outcome_nodes = [n for n in nodes if 'outcome' in n.lower()]
                mediator_nodes = [n for n in nodes if 'mediator' in n.lower() or 'mediat' in n.lower()]
                moderator_nodes = [n for n in nodes if 'moderator' in n.lower() or 'moderat' in n.lower()]
                confounder_nodes = [n for n in nodes if 'confounder' in n.lower() or 'confound' in n.lower()]
                collider_nodes = [n for n in nodes if 'collider' in n.lower() or 'collid' in n.lower()]

                # Assign nodes to layers based on topological order
                # Layer 0: Treatment/Source nodes (no predecessors or explicitly named)
                # Layer 1: Confounders (affect both treatment and outcome)
                # Layer 2: Mediators/Moderators (between treatment and outcome)
                # Layer 3: Colliders (affected by multiple nodes)
                # Layer 4: Outcomes/Sink nodes (no successors or explicitly named)

                layers = {0: [], 1: [], 2: [], 3: [], 4: []}

                # First, place explicitly named nodes
                for node in treatment_nodes:
                    layers[0].append(node)
                for node in outcome_nodes:
                    layers[4].append(node)
                for node in confounder_nodes:
                    layers[1].append(node)
                for node in mediator_nodes + moderator_nodes:
                    layers[2].append(node)
                for node in collider_nodes:
                    layers[3].append(node)

                # For unnamed nodes, use graph structure
                assigned = set(treatment_nodes + outcome_nodes + confounder_nodes +
                             mediator_nodes + moderator_nodes + collider_nodes)

                for node in nodes:
                    if node in assigned:
                        continue

                    in_deg = G.in_degree(node)
                    out_deg = G.out_degree(node)

                    # Source nodes (no predecessors)
                    if in_deg == 0:
                        layers[0].append(node)
                    # Sink nodes (no successors)
                    elif out_deg == 0:
                        layers[4].append(node)
                    # Intermediate nodes - use topological depth
                    else:
                        # Calculate shortest path from sources
                        sources = [n for n in G.nodes() if G.in_degree(n) == 0]
                        if sources:
                            try:
                                depths = []
                                for source in sources:
                                    if nx.has_path(G, source, node):
                                        depth = nx.shortest_path_length(G, source, node)
                                        depths.append(depth)

                                if depths:
                                    avg_depth = sum(depths) / len(depths)
                                    # Map depth to layer (1-3)
                                    if avg_depth <= 1:
                                        layers[1].append(node)
                                    elif avg_depth <= 2:
                                        layers[2].append(node)
                                    else:
                                        layers[3].append(node)
                                else:
                                    layers[2].append(node)
                            except:
                                layers[2].append(node)
                        else:
                            layers[2].append(node)

                # Create positions with proper spacing
                pos = {}
                x_spacing = 1.0  # Horizontal spacing between layers
                y_spacing = 0.8  # Vertical spacing between nodes in same layer

                for layer_idx, layer_nodes in layers.items():
                    if not layer_nodes:
                        continue

                    x = layer_idx * x_spacing

                    # Center nodes vertically
                    for i, node in enumerate(layer_nodes):
                        y = (len(layer_nodes) - 1) / 2.0 - i
                        pos[node] = (x, y * y_spacing)

                print("Using enhanced custom hierarchical layout")

            except Exception as e:
                # Final fallback to spring layout
                import traceback
                traceback.print_exc()
                print(f"Warning: Custom hierarchical layout failed ({e}), using spring layout")
                pos = nx.spring_layout(G, k=2, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G, k=2, iterations=50)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Get colors
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        node_color = color_map(0.6)
        edge_color = color_map(0.3)
        highlight_color = color_map(0.9)
    else:
        node_color = 'steelblue'
        edge_color = 'gray'
        highlight_color = get_color('highlight', style_mode)

    # Draw edges (with arrows ending at node edges)
    # Calculate node radius based on node size
    node_radius = 0.15  # Approximate radius for size 3000-3500 nodes

    regular_edges = [e for e in G.edges() if not (highlight_edges and e in highlight_edges)]
    if regular_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=regular_edges, edge_color=edge_color,
            width=2, alpha=0.6, arrows=True, arrowsize=20,
            arrowstyle='->', ax=ax,
            connectionstyle='arc3,rad=0.1',  # Slight curve to avoid overlap
            node_size=3000, min_source_margin=15, min_target_margin=15
        )

    # Draw highlighted edges
    if highlight_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=highlight_edges, edge_color=get_color('warning', style_mode),
            width=3, alpha=0.8, arrows=True, arrowsize=20,
            arrowstyle='->', ax=ax,
            connectionstyle='arc3,rad=0.1',
            node_size=3500, min_source_margin=15, min_target_margin=15
        )

    # Draw nodes
    regular_nodes = [n for n in G.nodes() if not (highlight_nodes and n in highlight_nodes)]
    if regular_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=regular_nodes, node_color=[node_color] * len(regular_nodes),
            node_size=3000, alpha=0.8, edgecolors='black', linewidths=2, ax=ax
        )

    # Draw highlighted nodes
    if highlight_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=highlight_nodes, node_color=[highlight_color] * len(highlight_nodes),
            node_size=3500, alpha=0.9, edgecolors='black', linewidths=3, ax=ax
        )

    # Draw labels with white-bordered backgrounds for readability
    for node in G.nodes():
        x, y = pos[node]
        ax.text(x, y, node,
               fontsize=11,
               fontweight='bold',
               color='#111111',
               ha='center',
               va='center',
               bbox=dict(facecolor='white', edgecolor='#333333', linewidth=1.5,
                        boxstyle='round,pad=0.5', alpha=0.95),
               zorder=10)

    ax.axis('off')
    ax.margins(0.15)

    # Title
    if title or subtitle:
        apply_titles(fig, title=title, subtitle=subtitle)

    # Layout
    fig.tight_layout(rect=(0, 0, 1, 0.94 if subtitle or title else 0.98))

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def dag_from_formula(
    formula: str,
    layout: str = "hierarchical",
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "viridis",
    figsize: tuple = (10, 8),
    output_path: Optional[str] = None,
):
    """
    Create a DAG from a formula string.

    Parameters
    ----------
    formula : str
        Formula describing edges, e.g., "A -> B; A -> C; C -> B".
    layout : str, default "hierarchical"
        Layout algorithm.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "viridis"
        Color scheme.
    figsize : tuple, default (10, 8)
        Figure size.
    output_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> dag_from_formula("Treatment -> Outcome; Confounder -> Treatment; Confounder -> Outcome",
    ...                  title='DAG from Formula')
    """
    # Parse formula
    edges = []
    for edge_str in formula.split(';'):
        edge_str = edge_str.strip()
        if '->' in edge_str:
            source, target = edge_str.split('->')
            edges.append((source.strip(), target.strip()))

    # Extract nodes
    nodes = list(set([e[0] for e in edges] + [e[1] for e in edges]))

    return dag(
        nodes=nodes,
        edges=edges,
        layout=layout,
        title=title,
        subtitle=subtitle,
        style_mode=style_mode,
        figsize=figsize,
        output_path=output_path
    )


# ==============================================================================
# INTERACTIVE VERSION
# ==============================================================================

def dag_interactive(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    layout: str = "spring",
    highlight_nodes: Optional[List[str]] = None,
    highlight_edges: Optional[List[Tuple[str, str]]] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "viridis",
):
    """
    Interactive DAG using Plotly.

    Parameters
    ----------
    nodes : list of str
        List of node names.
    edges : list of tuples
        List of directed edges.
    layout : str, default "spring"
        Layout algorithm.
    highlight_nodes : list of str, optional
        Nodes to highlight.
    highlight_edges : list of tuples, optional
        Edges to highlight.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "viridis"
        Color scheme.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> dag_interactive(nodes, edges, highlight_nodes=['Treatment', 'Outcome'],
    ...                 title='Interactive Causal DAG')
    """
    if nx is None:
        raise ImportError("networkx is required. Install with: pip install networkx")

    set_style(style_mode)

    # Create graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Get layout with improved hierarchical positioning (same as static version)
    if layout == "hierarchical":
        # Try graphviz dot layout first (best for DAGs)
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR')
            # Normalize positions for better display
            if pos:
                x_coords = [p[0] for p in pos.values()]
                y_coords = [p[1] for p in pos.values()]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Normalize to reasonable range
                for node in pos:
                    x, y = pos[node]
                    pos[node] = (
                        (x - x_min) / (x_max - x_min + 0.001) * 3,
                        (y - y_min) / (y_max - y_min + 0.001) * 2
                    )
                print("Using graphviz dot layout for optimal DAG visualization")
        except:
            # Fallback to custom hierarchical layout
            try:
                # Enhanced column-based layout with topological ordering
                treatment_nodes = [n for n in nodes if 'treatment' in n.lower()]
                outcome_nodes = [n for n in nodes if 'outcome' in n.lower()]
                mediator_nodes = [n for n in nodes if 'mediator' in n.lower() or 'mediat' in n.lower()]
                moderator_nodes = [n for n in nodes if 'moderator' in n.lower() or 'moderat' in n.lower()]
                confounder_nodes = [n for n in nodes if 'confounder' in n.lower() or 'confound' in n.lower()]
                collider_nodes = [n for n in nodes if 'collider' in n.lower() or 'collid' in n.lower()]

                # Assign nodes to layers
                layers = {0: [], 1: [], 2: [], 3: [], 4: []}

                # Place explicitly named nodes
                for node in treatment_nodes:
                    layers[0].append(node)
                for node in outcome_nodes:
                    layers[4].append(node)
                for node in confounder_nodes:
                    layers[1].append(node)
                for node in mediator_nodes + moderator_nodes:
                    layers[2].append(node)
                for node in collider_nodes:
                    layers[3].append(node)

                # For unnamed nodes, use graph structure
                assigned = set(treatment_nodes + outcome_nodes + confounder_nodes +
                             mediator_nodes + moderator_nodes + collider_nodes)

                for node in nodes:
                    if node in assigned:
                        continue

                    in_deg = G.in_degree(node)
                    out_deg = G.out_degree(node)

                    if in_deg == 0:
                        layers[0].append(node)
                    elif out_deg == 0:
                        layers[4].append(node)
                    else:
                        sources = [n for n in G.nodes() if G.in_degree(n) == 0]
                        if sources:
                            try:
                                depths = []
                                for source in sources:
                                    if nx.has_path(G, source, node):
                                        depth = nx.shortest_path_length(G, source, node)
                                        depths.append(depth)

                                if depths:
                                    avg_depth = sum(depths) / len(depths)
                                    if avg_depth <= 1:
                                        layers[1].append(node)
                                    elif avg_depth <= 2:
                                        layers[2].append(node)
                                    else:
                                        layers[3].append(node)
                                else:
                                    layers[2].append(node)
                            except:
                                layers[2].append(node)
                        else:
                            layers[2].append(node)

                # Create positions
                pos = {}
                x_spacing = 1.0
                y_spacing = 0.8

                for layer_idx, layer_nodes in layers.items():
                    if not layer_nodes:
                        continue
                    x = layer_idx * x_spacing
                    for i, node in enumerate(layer_nodes):
                        y = (len(layer_nodes) - 1) / 2.0 - i
                        pos[node] = (x, y * y_spacing)

                print("Using enhanced custom hierarchical layout")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Warning: Custom hierarchical layout failed ({e}), using spring layout")
                pos = nx.spring_layout(G, k=2, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50)
    else:
        pos = nx.spring_layout(G, k=2, iterations=50)

    # Create edge traces
    edge_traces = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        is_highlighted = highlight_edges and edge in highlight_edges

        highlight_edge_color = get_color('warning', style_mode)
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=3 if is_highlighted else 2,
                color=highlight_edge_color if is_highlighted else 'gray'
            ),
            hoverinfo='none',
            showlegend=False,
        )
        edge_traces.append(edge_trace)

    # Create node trace
    node_x = []
    node_y = []
    node_colors = []
    node_text = []

    highlight_node_color = get_color('highlight', style_mode)
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        is_highlighted = highlight_nodes and node in highlight_nodes
        node_colors.append(highlight_node_color if is_highlighted else 'steelblue')

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        marker=dict(
            size=50,
            color=node_colors,
            line=dict(width=2, color='black')
        ),
        hoverinfo='text',
        hovertext=node_text,
        showlegend=False,
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    # Add text annotations with white backgrounds for better readability
    for i, node in enumerate(G.nodes()):
        x, y = pos[node]
        fig.add_annotation(
            x=x,
            y=y,
            text=node,
            showarrow=False,
            font=dict(size=13, color='black', family='Arial Black'),
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#333333',
            borderwidth=1.5,
            borderpad=4,
        )

    # Layout
    title_text = f"<b>{title or 'Causal DAG'}</b>"
    if subtitle:
        title_text += f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>"

    fig.update_layout(
        template="plotly_white",
        height=700,
        margin=dict(t=90, b=30, l=30, r=30),
        title=dict(text=title_text, x=0.5, xanchor="center"),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
    )

    return fig
