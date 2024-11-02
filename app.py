import streamlit as st
import pandas as pd
import json

import networkx as nx
import matplotlib.pyplot as plt
import igraph as ig

from app_functions import create_graph
from app_functions import draw_one_graph_streamlit
from app_functions import draw_circle_graph_streamlit
from app_functions import draw_arc_graph
from app_functions import get_degree
from app_functions import draw_degree_rank
from app_functions import draw_degree_hist
from app_functions import describe_ave_degree
from app_functions import report_eccentricity
from app_functions import get_top10_degree_centrality
from app_functions import get_top10_betweenness_centrality
from app_functions import plot_betweenness
from app_functions import get_top10_pagerank
from app_functions import corr_measures
from app_functions import detect_communities_girvan_newman
from app_functions import analyze_communities
from app_functions import visualize_partition_interactive
from app_functions import visualize_partition_interactive_KL

st.set_page_config(
    page_title="StarWars character analysis",
    layout="wide"
)

def sidebar_nav():
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Go to",
        ["Welcome", 
         "EDA", 
         "Degree Analysis", 
         "Eccentricity Analysis", 
         "Centrality Analysis", 
         "Corralation Analysis", 
         "Community Detection",
         "Conclusion", 
         "Dataset"]
    )

def welcome_page():
    st.markdown("# StarWars Character importance Analysis")
    
    st.write("### Explore Character Interactions by Episode")
    
    episode_options = {
        "Episode I: The Phantom Menace": 1,
        "Episode II: Attack of the Clones": 2,
        "Episode III: Revenge of the Sith": 3,
        "Episode IV: A New Hope": 4,
        "Episode V: The Empire Strikes Back": 5,
        "Episode VI: Return of the Jedi": 6,
        "Episode VII: The Force Awakens": 7,
        "Total Season": 8
    }
    
    selected_episode = st.selectbox(
        "Select Star Wars Episode",
        options=list(episode_options.keys())
    )
    
    episode_num = str(episode_options[selected_episode])
    st.session_state.episode_num = episode_num
    
    st.write("""
    Welcome to the Star Wars Character Network Analysis Dashboard! This interactive tool explores 
    the complex web of relationships between characters in the Star Wars universe, revealing the 
    hidden importance and connections of your favorite characters.
    
    ### What This Dashboard Offers:
    
    📈 **Exploratory Data Analysis (EDA)**
    - Discover the overall structure of character interactions
    - Identify key patterns in character relationships
    - Visualize the density of connections across different episodes
    
    🌟 **Character Importance Metrics**
    - **Degree Analysis**: Find out which characters have the most direct connections
    - **Eccentricity Analysis**: Understand how central or peripheral characters are in the network
    - **Centrality Analysis**: Discover the most influential characters through various measures:
        - Who are the key connectors?
        - Which characters bridge different groups?
        - Who has the most strategic position in the network?
    
    📊 **Dataset Overview**

    ### Done By:
    - Aditya        20pd02
    - Kartheepan G  20pd11
    - Kesavan G     20pd13
    """)

    if episode_num == str(8):
        file_interaction = 'starwars-full-interactions.json'
        file_mention = 'starwars-full-mentions.json'
        file_all = 'starwars-full-interactions-allCharacters.json'
    else:
        file_interaction = 'starwars-episode-' + episode_num + '-interactions.json'
        file_mention = 'starwars-episode-' + episode_num + '-mentions.json'
        file_all = 'starwars-episode-' + episode_num + '-interactions-allCharacters.json'

    with open(file_interaction) as f: data_interactions = json.load(f)
    with open(file_mention) as f: data_mentions = json.load(f)
    with open(file_all) as f: data_all = json.load(f)

    G_interactions = create_graph(data_interactions)
    G_mentions = create_graph(data_mentions)
    G_all = create_graph(data_all)
    st.session_state.G = [G_interactions, G_mentions, G_all]

def eda_page():
    G = st.session_state.G
    net = []
    net.append(ig.Graph.from_networkx(G[0]))
    net.append(ig.Graph.from_networkx(G[1]))
    net.append(ig.Graph.from_networkx(G[2]))

    st.markdown("# Exploratory Data Analysis")
    st.markdown("## Network Details")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Data from the Intereactions Network")
        st.write("Number of Nodes:", G[0].number_of_nodes())
        st.write("Number of Edges:", G[0].number_of_edges())
        st.write("")
    with col2:
        st.markdown('#### Data from the Mentions Network')
        st.write('Number of nodes: ', G[1].number_of_nodes())
        st.write('Number of links:', G[1].number_of_edges())
        st.write("")
    with col3:
        st.markdown('#### Data from the Merge Network')
        st.write('Number of nodes: ', G[2].number_of_nodes())
        st.write('Number of links:', G[2].number_of_edges())
        st.write("")

    st.write("Nodes in Merge Network but not in Interactino Network: ", set(G[1].nodes()) - set(G[0].nodes()))
    st.write("Extra number of edges in Merge Network than in Interaction Network: ", G[2].number_of_edges()-G[0].number_of_edges())

    st.markdown("## Data Visualization")

    plot_options = ["Interaction Plots", "Mention Plots", "Merge Plots"]
    selected_plot = st.selectbox("Select a plot type:", plot_options)

    if selected_plot == "Interaction Plots":
        st.markdown("### Interaction Network")
        draw_one_graph_streamlit(G[0], 'Interactions Network')
        draw_circle_graph_streamlit(G[0], 'Circle Graph')
        draw_arc_graph(G[0], 'Arc Network')
    
    if selected_plot == "Mention Plots":
        st.markdown("### Mention Network")
        draw_one_graph_streamlit(G[1], 'Mention Network')
        draw_circle_graph_streamlit(G[1], 'Circle Graph')
        draw_arc_graph(G[1], 'Arc Network')
    
    if selected_plot == "Merge Plots":
        st.markdown("### Merge Network")
        draw_one_graph_streamlit(G[2], 'Merge Network')
        draw_circle_graph_streamlit(G[2], 'Circle Graph')
        draw_arc_graph(G[2], 'Arc Network')
    
    st.session_state.net = net

def degree_analysis():
    G = st.session_state.G
    st.markdown("# Degree Analysis")
    
    plot_options = ["Interaction Plots", "Mention Plots", "Merge Plots"]
    selected_plot = st.selectbox("Select a plot type:", plot_options)

    if selected_plot == 'Interaction Plots':
        degree_data = get_degree(G[0], sort=True)[:10]
        st.markdown("## Degree data")
        col1, col2 = st.columns(2)
        with col1:
            df = pd.DataFrame(degree_data, columns=["Character", "Degree"])
            st.write("Let's use the top 10 characters with the highest degree (using the weighted metric) to analyze whether there is a difference between the 3 networks :\n", df)
        with col2:
            st.write("Let's calculate the average degree of the networks and then check who are the characters with degree greater than the average degree.")
            describe_ave_degree(G[0])
        
        draw_degree_rank(G[0], 'Interactions Netowork')
        draw_degree_hist(G[0], 'Interactions Netowork')
        draw_degree_rank(G[0], 'Interactions Netowork', weighted=True)
        draw_degree_hist(G[0], 'Interactions Netowork', weighted=True)
    
    if selected_plot == 'Mention Plots':
        degree_data = get_degree(G[1], sort=True)[:10]
        st.markdown("## Degree data")
        col1, col2= st.columns(2)
        with col1:
            df = pd.DataFrame(degree_data, columns=["Character", "Degree"])
            st.write("Let's use the top 10 characters with the highest degree (using the weighted metric) to analyze whether there is a difference between the 3 networks :\n", df)
        with col2:
            st.write("Let's calculate the average degree of the networks and then check who are the characters with degree greater than the average degree.")
            describe_ave_degree(G[1])
        draw_degree_rank(G[1], 'Interactions Netowork')
        draw_degree_hist(G[1], 'Interactions Netowork')
        draw_degree_rank(G[1], 'Interactions Netowork', weighted=True)
        draw_degree_hist(G[1], 'Interactions Netowork', weighted=True)
    
    if selected_plot == 'Merge Plots':
        degree_data = get_degree(G[2], sort=True)[:10]
        st.markdown("## Degree data")
        col1, col2= st.columns(2)
        with col1:
            df = pd.DataFrame(degree_data, columns=["Character", "Degree"])
            st.write("Let's use the top 10 characters with the highest degree (using the weighted metric) to analyze whether there is a difference between the 3 networks :\n", df)
        with col2:
            st.write("Let's calculate the average degree of the networks and then check who are the characters with degree greater than the average degree.")
            describe_ave_degree(G[2])
        draw_degree_rank(G[2], 'Interactions Netowork')
        draw_degree_hist(G[2], 'Interactions Netowork')
        draw_degree_rank(G[2], 'Interactions Netowork', weighted=True)
        draw_degree_hist(G[2], 'Interactions Netowork', weighted=True)


def eccentricity_analysis():
    G = st.session_state.G
    st.markdown("# Eccentricity Analysis")
    col1, col2, col3 = st.columns(3)
    with col1: report_eccentricity(G[0], '### Interactions Network')
    with col2: report_eccentricity(G[1], '### Mentions Network')
    with col3: report_eccentricity(G[2], '### All Characters Network')


def centrality_analysis():
    G = st.session_state.G
    st.markdown("# Centrality Analysis")
    st.markdown("## Degree Centrality")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Interaction Network")
        centrality_data = get_top10_degree_centrality(G[0])
        df = pd.DataFrame(centrality_data)
        st.dataframe(df)  

    with col2:
        st.markdown("### Mention Network")
        centrality_data = get_top10_degree_centrality(G[1])
        df = pd.DataFrame(centrality_data)
        st.dataframe(df)

    with col3:
        st.markdown("### Merge Network")
        centrality_data = get_top10_degree_centrality(G[2])
        df = pd.DataFrame(centrality_data)
        st.dataframe(df)
    
    st.markdown("## Betweenness Centrality")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Interaction Network")
        centrality_data = get_top10_betweenness_centrality(G[0])
        df = pd.DataFrame(centrality_data)
        st.dataframe(df)  

    with col2:
        st.markdown("### Mention Network")
        centrality_data = get_top10_betweenness_centrality(G[1])
        df = pd.DataFrame(centrality_data)
        st.dataframe(df)

    with col3:
        st.markdown("### Merge Network")
        centrality_data = get_top10_betweenness_centrality(G[2])
        df = pd.DataFrame(centrality_data)
        st.dataframe(df)

    st.markdown("## Betweenness Centrality with Edge Weight")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Interaction Network")
        centrality_data = get_top10_betweenness_centrality(G[0], True)
        df = pd.DataFrame(centrality_data)
        st.dataframe(df)  

    with col2:
        st.markdown("### Mention Network")
        centrality_data = get_top10_betweenness_centrality(G[1], True)
        df = pd.DataFrame(centrality_data)
        st.dataframe(df)

    with col3:
        st.markdown("### Merge Network")
        centrality_data = get_top10_betweenness_centrality(G[2], True)
        df = pd.DataFrame(centrality_data)
        st.dataframe(df)
    

    st.markdown("## Betweenness Centrality Plot")
    net = st.session_state.net
    # st.write("Let's visualize the vertex and edge betweenness using the **igraph** library. We will use the betweenness centrality of a node the betweenness centrality of an edge.")
    fig, axs = plt.subplots(
        3, 3,
        figsize=(20, 10),
        gridspec_kw=dict(height_ratios=(20, 1, 1)),
    )
    plot_betweenness(net[0], net[0].betweenness(), net[0].edge_betweenness(), *axs[:, 0])
    plot_betweenness(net[1], net[1].betweenness(), net[1].edge_betweenness(), *axs[:, 1])
    plot_betweenness(net[2], net[2].betweenness(), net[2].edge_betweenness(), *axs[:, 2])
    fig.tight_layout(h_pad=1)
    st.pyplot(fig)

    st.markdown("## PageRank")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Interaction Network")
        centrality_data = get_top10_pagerank(G[0])
        df = pd.DataFrame(centrality_data)
        st.dataframe(df)  

    with col2:
        st.markdown("### Mention Network")
        centrality_data = get_top10_pagerank(G[1])
        df = pd.DataFrame(centrality_data)
        st.dataframe(df)

    with col3:
        st.markdown("### Merge Network")
        centrality_data = get_top10_pagerank(G[2])
        df = pd.DataFrame(centrality_data)
        st.dataframe(df)

def correlation_page():
    G = st.session_state.G

    correlations = []
    st.markdown("### Interaction Network")
    correlations.append(corr_measures(G[0]))
    st.write("Correlation Matrix:")
    st.dataframe(correlations[0].T.corr())

    st.markdown("### Mention Network")
    correlations.append(corr_measures(G[1]))
    st.write("Correlation Matrix:")
    st.dataframe(correlations[1].T.corr())

    st.markdown("### Merge Network")
    correlations.append(corr_measures(G[2]))
    st.write("Correlation Matrix:")
    st.dataframe(correlations[2].T.corr())

    st.session_state.correlations = correlations

def community_detection():
    G = st.session_state.G
    st.markdown("# Community Detection")

    st.markdown("### Girvan-Newman Algorithm")
    col1, col2, col3 = st.columns(3)
    with col1:
        gn_communities, gn_modularity = detect_communities_girvan_newman(G[0])
        gn_analysis = analyze_communities(G[0], gn_communities)
        for comm, size in gn_analysis['community_sizes'].items(): st.write(f"Community {comm+1}:", size)
        fig = visualize_partition_interactive(G[0], gn_communities)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gn_communities, gn_modularity = detect_communities_girvan_newman(G[1])
        gn_analysis = analyze_communities(G[1], gn_communities)
        for comm, size in gn_analysis['community_sizes'].items(): st.write(f"Community {comm+1}:", size)
        fig = visualize_partition_interactive(G[1], gn_communities)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        gn_communities, gn_modularity = detect_communities_girvan_newman(G[2])
        gn_analysis = analyze_communities(G[2], gn_communities)
        for comm, size in gn_analysis['community_sizes'].items(): st.write(f"Community {comm+1}:", size)
        fig = visualize_partition_interactive(G[2], gn_communities)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Kernighan-Lin")
    col1, col2, col3 = st.columns(3)
    with col1:
        partition = nx.algorithms.community.kernighan_lin_bisection(G[0])
        st.write("Community 1:", len(partition[0]))
        st.write("Community 2:", len(partition[1]))
        
        fig = visualize_partition_interactive_KL(G[0], partition)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        partition = nx.algorithms.community.kernighan_lin_bisection(G[1])
        st.write("Community 1:", len(partition[0]))
        st.write("Community 2:", len(partition[1]))
        
        fig = visualize_partition_interactive_KL(G[1], partition)
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        partition = nx.algorithms.community.kernighan_lin_bisection(G[2])
        st.write("Community 1:", len(partition[0]))
        st.write("Community 2:", len(partition[1]))
        
        fig = visualize_partition_interactive_KL(G[2], partition)
        st.plotly_chart(fig, use_container_width=True)

def dataset_page():
    st.markdown("# Dataset")
    episode_num = st.session_state.episode_num

    if episode_num == str(8):
        file_interaction = 'starwars-full-interactions.json'
        file_mention = 'starwars-full-mentions.json'
        file_all = 'starwars-full-interactions-allCharacters.json'
    else:
        file_interaction = 'starwars-episode-' + episode_num + '-interactions.json'
        file_mention = 'starwars-episode-' + episode_num + '-mentions.json'
        file_all = 'starwars-episode-' + episode_num + '-interactions-allCharacters.json'

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Interaction Network Data")
        with open(file_interaction) as f: data_interactions = json.load(f)
        st.json(data_interactions)
    with col2: 
        st.markdown("### Mention Network Data")
        with open(file_mention) as f: data_mention = json.load(f)
        st.json(data_mention)
    with col3:
        st.markdown("### Merge Network Data")
        with open(file_all) as f: data_merge = json.load(f)
        st.json(data_merge)
    

def conclusion_page():
    correlations = st.session_state.correlations
    st.markdown("# Conclusion")

    st.markdown("## PageRank")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Interaction Network")
        max_values = correlations[0].idxmax(axis=1)
        df_max = pd.DataFrame(max_values, columns=["Node"]).reset_index()
        df_max.columns = ["Centrality Measure", "Node"]
        st.write("Node with Maximum Value for Each Centrality Measure:")
        st.dataframe(df_max)
    
    with col2:
        st.markdown("### Mention Network")
        max_values = correlations[1].idxmax(axis=1)
        df_max = pd.DataFrame(max_values, columns=["Node"]).reset_index()
        df_max.columns = ["Centrality Measure", "Node"]
        st.write("Node with Maximum Value for Each Centrality Measure:")
        st.dataframe(df_max)
    
    with col3:
        st.markdown("### Merge Network")
        max_values = correlations[2].idxmax(axis=1)
        df_max = pd.DataFrame(max_values, columns=["Node"]).reset_index()
        df_max.columns = ["Centrality Measure", "Node"]
        st.write("Node with Maximum Value for Each Centrality Measure:")
        st.dataframe(df_max)

def main():
    current_page = sidebar_nav()
    
    if current_page == "Welcome":
        welcome_page()
    elif current_page == "EDA":
        eda_page()
    elif current_page == "Degree Analysis":
        degree_analysis()
    elif current_page == "Eccentricity Analysis":
        eccentricity_analysis()
    elif current_page == "Centrality Analysis":
        centrality_analysis()
    elif current_page == "Corralation Analysis":
        correlation_page()
    elif current_page == "Community Detection":
        community_detection()
    elif current_page == "Dataset":
        dataset_page()
    else:
        conclusion_page()

if __name__ == "__main__":
    main()
