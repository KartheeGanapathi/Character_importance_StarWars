import streamlit as st

import numpy as np
import pandas as pd

import networkx as nx
import nxviz as nv
from nxviz import annotate
from nxviz import nodes
from nxviz import edges
from nxviz.plots import aspect_equal, despine

import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import igraph as ig

from pyvis.network import Network
import plotly.graph_objects as go

from collections import defaultdict

# def kernighan_lin_partition(G):
#     # Perform the Kernighan-Lin algorithm and partition the graph into two communities
#     partition = nx.algorithms.community.kernighan_lin_bisection(G)
#     return partition

def visualize_partition_interactive_KL(G, partition):
    pos = nx.spring_layout(G, seed=42)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    colors = ['skyblue', 'lightgreen']
    node_traces = []

    for i, community in enumerate(partition):
        node_x = []
        node_y = []
        node_text = []

        for node in community:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f'{node}')

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=False,
                color=colors[i],
                size=15,
                line_width=2),
            textposition="bottom center"
        )
        node_traces.append(node_trace)

    fig = go.Figure(data=[edge_trace] + node_traces,
                    layout=go.Layout(
                        title='Kernighan-Lin Community Detection',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        annotations=[dict(
                            text="Interactive community detection with Plotly",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002
                        )],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )
    return fig

def detect_communities_girvan_newman(G):
    comp = nx.community.girvan_newman(G)
    communities_list = list(next(comp))
    
    communities = {}
    for i, comm in enumerate(communities_list):
        for node in comm:
            communities[node] = i
    
    modularity = compute_modularity(G, communities)
    
    return communities, modularity

def compute_modularity(G, communities):
    m = G.number_of_edges()
    degrees = dict(G.degree())
    Q = 0
    for node1, node2 in G.edges():
        if communities[node1] == communities[node2]:
            Q += 1 - (degrees[node1] * degrees[node2]) / (2 * m)
    return Q / (2 * m)

def visualize_partition_interactive(G, communities):
    pos = nx.spring_layout(G, seed=42)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    n_communities = len(set(communities.values()))
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'violet', 'orange', 'cyan'][:n_communities]
    node_traces = []

    for i in range(n_communities):
        node_x = []
        node_y = []
        node_text = []

        for node in G.nodes():
            if communities[node] == i:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(f'{node}')

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=False,
                color=colors[i],
                size=15,
                line_width=2),
            textposition="bottom center"
        )
        node_traces.append(node_trace)

    fig = go.Figure(data=[edge_trace] + node_traces,
                    layout=go.Layout(
                        title='Girvan-Newman Community Detection',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        annotations=[dict(
                            text="Interactive community detection with Plotly",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002
                        )],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )
    return fig

def analyze_communities(G, communities):
    analysis = {
        'n_communities': len(set(communities.values())),
        'community_sizes': defaultdict(int),
        'community_density': {},
        'community_centrality': {}
    }
    
    for node, comm in communities.items():
        analysis['community_sizes'][comm] += 1
    
    for comm_id in set(communities.values()):
        comm_nodes = [node for node in G.nodes() if communities[node] == comm_id]
        subgraph = G.subgraph(comm_nodes)
        
        analysis['community_density'][comm_id] = nx.density(subgraph)
        
        if len(comm_nodes) > 1:
            centrality = nx.betweenness_centrality(subgraph)
            analysis['community_centrality'][comm_id] = np.mean(list(centrality.values()))
        else:
            analysis['community_centrality'][comm_id] = 0
    
    return analysis

def draw_degree_rank(G, title, weighted=False):
    degrees = get_degree(G, sort=False, weighted=weighted)
    degree_sequence = sorted(list(degrees.values()), reverse=True)
    dmax = max(degree_sequence)

    degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    people = [degrees[x][0] for x in np.arange(0, len(degrees))]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=people, 
        y=degree_sequence, 
        mode='lines+markers', 
        marker=dict(color='blue', size=8), 
        line=dict(color='blue'),
        hoverinfo='text',
        text=[f"{person}: {deg}" for person, deg in zip(people, degree_sequence)]
    ))

    fig.update_layout(
        title=f"Degree Rank Plot - {title}",
        xaxis_title="Characters",
        yaxis_title="Degree",
        xaxis_tickangle=-80,
        height=600,
        width=1000,
        margin=dict(l=40, r=40, t=40, b=120)
    )

    st.plotly_chart(fig, use_container_width=True)

def draw_one_graph_streamlit(G, title):
    pos = nx.spring_layout(G, seed=42)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(node) for node in G.nodes()],
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGn',
            size=10,
            color='green',
            line_width=2)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         title=title,
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=0, l=0, r=0, t=40),
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                     )
    st.plotly_chart(fig)

def draw_circle_graph_streamlit(G, title):
    pos = nx.circular_layout(G)  
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(node) for node in G.nodes()],
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGn',
            size=10,
            color='green',
            line_width=2)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         title=title,
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=0, l=0, r=0, t=40),
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                     )
    st.plotly_chart(fig)

def create_graph(data):
    G = nx.Graph()

    for node in data['nodes']:
        G.add_node(node['name'])
        G.nodes[node['name']]['colour'] = node['colour']
        G.nodes[node['name']]['scenes'] = node['value']
        G.nodes[node['name']]['name'] = node['name']
    
    for edge in data['links']:
        G.add_edge(data['nodes'][edge['source']]['name'], data['nodes'][edge['target']]['name'], weight=edge['value'])
    
    return G
    
def draw_arc_graph(G, title, size=14):
    st.write("")
    fig, ax = plt.subplots(figsize=(size, size))
    try:
        pos = nodes.arc(G, group_by="name", color_by="colour")
        edges.arc(G, pos)
        annotate.arc_group(G, group_by="name")
        plt.title(title, fontsize=20)
        despine()
        aspect_equal()
        st.pyplot(fig)
    except:
        st.write("Unable to plot arc graph as there are more colors (>12)")
    
def draw_graph_with_features(net, size=15):
    scaled_weight = ig.rescale(net.es["weight"], clamp=True)
    cmap2 = LinearSegmentedColormap.from_list("edge_cmap", ["lightblue", "midnightblue"])

    fig, ax = plt.subplots(figsize=(size,size))
    ig.plot(
        net,
        target=ax,
        layout="circle",
        vertex_size= [num/(max(net.vs["scenes"])*5) if num > 20 else 0.05 for num in net.vs["scenes"]], 
        vertex_color=[color for color in net.vs["colour"]],
        vertex_frame_width=4.0,
        vertex_frame_color="white",
        vertex_label=net.vs["name"],
        edge_width=net.es["weight"],
        edge_color= [cmap2(value) for value in scaled_weight]
    )

    st.pyplot(fig)

def get_degree(G, sort=False, weighted=False):
    degrees = {}
    for node in G.nodes():
        if weighted:
            degrees[node] = G.degree(weight='weight')[node]
        else:
            degrees[node] = G.degree[node]
    if sort:
        degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    return degrees

def draw_degree_hist(G, title, weighted=False):
    degrees = get_degree(G, sort=False, weighted=weighted)
    degree_sequence = sorted(list(degrees.values()), reverse=True)
    
    unique_degrees, degree_counts = np.unique(degree_sequence, return_counts=True)
    
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=unique_degrees,
        y=degree_counts,
        marker_color='blue',
        text=[f"Degree: {deg}<br>Count: {count}" for deg, count in zip(unique_degrees, degree_counts)],
        hoverinfo='text'
    ))

    fig.update_layout(
        title=f"Degree Histogram - {title}",
        xaxis_title="Degree",
        yaxis_title="# of Nodes",
        xaxis_tickmode='linear',
        xaxis_tick0=0,
        xaxis_dtick=1,
        height=600,
        width=1000,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

def describe_ave_degree(G):
    degrees = get_degree(G, sort=True)
    
    mean_degree = np.mean(list(dict(degrees).values()))
    
    characters = [(degrees[x][0], degrees[x][1]) for x in np.arange(0, len(degrees)) if degrees[x][1] >= mean_degree]
    df = pd.DataFrame(characters, columns=["Character", "Degree"])
    
    st.write(df)
    st.write('Number of characters: ', len(characters))
    st.write('Mean degree: ', mean_degree)

def report_eccentricity(G, net_name):
    st.markdown(net_name)
    if not nx.is_connected(G):
        st.write("The network is not closed. :(")
        return
    else:
        st.write('Diameter: ', nx.diameter(G))
        st.write('Radius: ', nx.radius(G))
        st.write('Center: ', nx.center(G)[0])

        characters = sorted(dict(nx.eccentricity(G)).items(), key=lambda x: x[1], reverse=False)
        df = pd.DataFrame(characters, columns=["Character", "Eccentricity"])
        
        st.write('The eccentricity of nodes in ascending order:')
        st.dataframe(df) 

def get_top10_degree_centrality(G):
    deg_cen = nx.degree_centrality(G)
    sorted_deg_cen = sorted(deg_cen.items(), key=lambda x: x[1], reverse=True)
    return sorted_deg_cen[:10]

def get_top10_betweenness_centrality(G, weighted=False):
    if weighted:
        betweenness = nx.betweenness_centrality(G, weight='weight')
    else:
        betweenness = nx.betweenness_centrality(G)
    nx.betweenness_centrality(G, weight='weight')
    sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    return sorted_betweenness[:10]

def plot_betweenness(g, vertex_betweenness, edge_betweenness, ax, cax1, cax2):
    scaled_vertex_betweenness = ig.rescale(vertex_betweenness, clamp=True)
    scaled_edge_betweenness = ig.rescale(edge_betweenness, clamp=True)
    cmap1 = LinearSegmentedColormap.from_list("vertex_cmap", ["pink", "indigo"])
    cmap2 = LinearSegmentedColormap.from_list("edge_cmap", ["lightblue", "midnightblue"])

    g.vs["color"] = [cmap1(betweenness) for betweenness in scaled_vertex_betweenness]
    g.vs["size"]  = ig.rescale(vertex_betweenness, (0.1, 0.5))
    g.es["color"] = [cmap2(betweenness) for betweenness in scaled_edge_betweenness]
    g.es["width"] = ig.rescale(edge_betweenness, (0.5, 1.0))

    ig.plot(
        g,
        target=ax,
        layout="fruchterman_reingold",
        vertex_frame_width=0.2,
        vertex_label=g.vs["name"],
        vertex_label_size=7.0
    )

    norm1 = ScalarMappable(norm=Normalize(0, max(vertex_betweenness)), cmap=cmap1)
    norm2 = ScalarMappable(norm=Normalize(0, max(edge_betweenness)), cmap=cmap2)
    plt.colorbar(norm1, cax=cax1, orientation="horizontal", label='Vertex Betweenness')
    plt.colorbar(norm2, cax=cax2, orientation="horizontal", label='Edge Betweenness')

def get_top10_pagerank(G):
    pr = nx.pagerank(G, alpha=0.9)
    sorted_pr_ep1 = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    return sorted_pr_ep1[:10]

def corr_measures(G):
    measures = [nx.degree_centrality(G),
                nx.betweenness_centrality(G), 
                nx.betweenness_centrality(G, weight='weight'), 
                nx.pagerank(G), 
            ]
    cor = pd.DataFrame.from_records(measures, index=['degree_centrality', 'betweenness_centrality',
                                                  'betweenness_centrality_weight', 'pagerank'])

    return cor
