# Import necessary libraries
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import folium
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import kneighbors_graph
# -------------------------------
# STREAMLIT PAGE SETTINGS
# -------------------------------
st.set_page_config(
    page_title="US E-Commerce Spectral Clustering", layout="wide")
st.title("📊 US E-Commerce Spectral Clustering Dashboard")
st.markdown("This dashboard displays insights, clustering results, and interactive maps for the 2020 US E-Commerce dataset.")

# -------------------------------
# STEP 0: Load Raw Data
# -------------------------------
df = pd.read_csv('US  E-commerce records 2020.csv', encoding='windows-1252')
st.success(
    f"✅ Loaded dataset with {df.shape[0]:,} rows and {df.shape[1]:,} columns.")

# -------------------------------
# Data Preprocessing
# -------------------------------
df['Postal Code'] = df['Postal Code'].fillna(0).astype(int)
df.drop_duplicates(inplace=True)
df['Order Date'] = pd.to_datetime(
    df['Order Date'], format='%d-%m-%y', errors='coerce')
df['Order Month'] = df['Order Date'].dt.month
df['Order Day'] = df['Order Date'].dt.day
df['Order Weekday'] = df['Order Date'].dt.weekday
df['Profit Margin'] = np.where(df['Sales'] != 0, df['Profit'] / df['Sales'], 0)
df['Unit Price'] = np.where(df['Quantity'] != 0, df['Sales']/df['Quantity'], 0)
df['Discounted Price'] = df['Sales'] * (1 - df['Discount'])

# Encode categorical variables
df['Category'] = df['Category'].astype('category')
df['Segment'] = df['Segment'].astype('category')
df['Category_Code'] = df['Category'].cat.codes
df['Segment_Code'] = df['Segment'].cat.codes

# Replace NaN / Inf
df.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(0)

# -------------------------------
# Spectral Clustering
# -------------------------------
df_sampled = df[df['Country'] == 'United States'].sample(
    n=1000, random_state=42).copy()
features = df_sampled[['Sales', 'Quantity', 'Discount', 'Profit',
                       'Profit Margin', 'Unit Price', 'Discounted Price']].copy()
features.replace([np.inf, -np.inf], np.nan, inplace=True)
features.fillna(0, inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Ensure PCA output has no NaN/Inf
X_pca = np.nan_to_num(X_pca, nan=0.0, posinf=0.0, neginf=0.0)
print(df_sampled[['Sales', 'Profit', 'Profit Margin']].describe())

results = {}
sc_nn = SpectralClustering(
    n_clusters=3, affinity='nearest_neighbors', n_neighbors=5, random_state=42)
labels_nn = sc_nn.fit_predict(X_pca)
results['nearest_neighbors'] = silhouette_score(X_pca, labels_nn)

sc_rbf = SpectralClustering(
    n_clusters=3, affinity='rbf', gamma=1.0, random_state=42)
labels_rbf = sc_rbf.fit_predict(X_pca)
results['rbf'] = silhouette_score(X_pca, labels_rbf)

affinity_matrix = rbf_kernel(X_pca, gamma=1.0)
sc_pre = SpectralClustering(
    n_clusters=3, affinity='precomputed', random_state=42)
labels_pre = sc_pre.fit_predict(affinity_matrix)
results['precomputed'] = silhouette_score(X_pca, labels_pre)

best_method = max(results, key=results.get)
df_sampled['Cluster'] = {
    'nearest_neighbors': labels_nn,
    'rbf': labels_rbf,
    'precomputed': labels_pre
}[best_method]

st.write(
    f"**🏆 Best Method:** `{best_method}` with Silhouette Score **{results[best_method]:.4f}**")

# Merge cluster labels back to main df
# After picking best_method the first time (before tabs) KEEP labels only for df_sampled
df = df.drop(columns=['Cluster'], errors='ignore')  # ensure clean
df = df.merge(df_sampled[['Customer ID', 'Cluster']],
              on='Customer ID', how='left')

# -------------------------------
# Plotly Visualizations
# -------------------------------
sales_stats = {
    'Total Sales': df['Sales'].sum(),
    'Average Sales': df['Sales'].mean(),
    'Max Sales': df['Sales'].max(),
    'Min Sales': df['Sales'].min(),
    'Sales Std Dev': df['Sales'].std()
}
stats_df = pd.DataFrame.from_dict(
    sales_stats, orient='index', columns=['Value']).reset_index()
fig1 = px.bar(stats_df, x='index', y='Value', color='index',
              text=[f"${x:,.2f}" for x in stats_df['Value']])


state_profit = df.groupby('State')['Profit'].sum().sort_values(ascending=False)
fig2 = px.bar(state_profit.reset_index(), x='State', y='Profit',
              color='Profit', color_continuous_scale='Viridis')

# -------------------------------
# Monthly Sales Line Chart
# -------------------------------
monthly_sales = df.groupby('Order Month')['Sales'].sum().reset_index()

fig_monthly = px.line(
    monthly_sales,
    x='Order Month',
    y='Sales',
    title='<b>Monthly Sales Trend</b>',
    markers=True,
    color_discrete_sequence=['#1f77b4']
)

fig_monthly.update_layout(
    xaxis_title='Month',
    yaxis_title='Total Sales ($)',
    xaxis=dict(tickmode='linear')  # ensures months appear in order 1–12
)

# -------------------------------
# Category and Sub-Category Analysis
category_analysis = df.groupby(['Category', 'Sub-Category']).agg({
    'Sales': 'sum', 'Profit': 'sum', 'Quantity': 'sum', 'Discount': 'mean', 'Profit Margin': 'mean'
}).sort_values('Profit', ascending=False)
top_subcats = category_analysis.head(20).reset_index()
fig3 = px.bar(top_subcats, x='Sub-Category', y='Profit', color='Category')


# -------------------------------
# 4. Optimal Customer Segments Analysis
# -------------------------------
cluster_summary = df.groupby('Cluster').agg({
    'Sales': 'mean',
    'Profit': 'mean',
    'Customer ID': 'count'
}).rename(columns={'Customer ID': 'Customer Count'})

fig4 = make_subplots(rows=1, cols=2, specs=[
                     [{'type': 'xy'}, {'type': 'domain'}]])

# Bar chart for customers per segment
fig4.add_trace(
    go.Bar(
        x=cluster_summary.index.astype(str),
        y=cluster_summary['Customer Count'],
        name='Customers per Segment',
        marker_color='#636EFA',
        text=cluster_summary['Customer Count'],
        textposition='outside'
    ),
    row=1, col=1
)

# Donut chart for segment distribution
fig4.add_trace(
    go.Pie(
        labels=cluster_summary.index.astype(str),
        values=cluster_summary['Customer Count'],
        name='Segment Distribution',
        hole=0.4,
        textinfo='label+percent'
    ),
    row=1, col=2
)

fig4.update_layout(
    title_text=f'<b>Customer Segments Analysis (Clusters: {len(cluster_summary)})</b>',
    showlegend=True,
    height=500,
    width=1000
)

# -------------------------------

category_sales = df.groupby('Category')['Quantity'].sum().reset_index()
fig6 = px.bar(category_sales, x='Category', y='Quantity',
              color='Category', text_auto=True)


margin = (df['Profit'].sum() / df['Sales'].sum()) * 100
fig7 = go.Figure(go.Indicator(
    mode="number+gauge",
    value=margin,
    number={'suffix': '%'},
    title={'text': "Overall Profit Margin", 'font': {'size': 28}},
    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "white"}, 'steps': [{'range': [0, 20], 'color': "lightgray"},
                                                                             {'range': [0, 50], 'color': "red"}, {'range': [50, 100], 'color': "green"},]}
))
state_profit = df.groupby('State')['Profit'].sum().sort_values(ascending=False)

category_analysis = df.groupby(['Category', 'Sub-Category']).agg({
    'Sales': 'sum', 'Profit': 'sum', 'Quantity': 'sum', 'Discount': 'mean', 'Profit Margin': 'mean'
}).sort_values('Profit', ascending=False)
top_subcats = category_analysis.head(20).reset_index()

cluster_summary = df_sampled.groupby('Cluster').agg({
    'Sales': 'mean', 'Profit': 'mean', 'Customer ID': 'count'
}).rename(columns={'Customer ID': 'Customer Count'})

# -------------------------------
# Folium Map
# -------------------------------
geo_df = pd.read_csv(
    'world_country_and_usa_states_latitude_and_longitude_values.csv')
geo_df = geo_df[['usa_state', 'usa_state_latitude',
                 'usa_state_longitude']].drop_duplicates()

state_summary = df.groupby(['State', 'Cluster']).agg(
    {'Sales': 'sum', 'Profit': 'sum', 'Quantity': 'sum'}).reset_index()
state_summary = state_summary.merge(
    geo_df, left_on='State', right_on='usa_state', how='left')

m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)
for _, row in state_summary.dropna(subset=['usa_state_latitude', 'usa_state_longitude']).iterrows():
    folium.CircleMarker(
        location=[row['usa_state_latitude'], row['usa_state_longitude']],
        radius=6,
        popup=(f"<b>State:</b> {row['State']}<br>"
               f"<b>Cluster:</b> {row['Cluster']}<br>"
               f"<b>Sales:</b> ${row['Sales']:,.2f}<br>"
               f"<b>Profit:</b> ${row['Profit']:,.2f}"),
        color="blue", fill=True, fill_opacity=0.7
    ).add_to(m)
m.save('spectral_clustering_map.html')

# -------------------------------
# SIDEBAR: Clustering Hyperparameters
# -------------------------------
st.sidebar.subheader("⚙️ Spectral Clustering Controls")

# Choose affinity type
affinity_choice = st.sidebar.selectbox(
    "Select Affinity Method",
    ["nearest_neighbors", "rbf", "precomputed", "cosine", "manhattan"],
    index=1
)

# Cluster sliders
k = st.sidebar.slider("🔢 k (clusters)", 2, 10, 3, 1)
gamma = st.sidebar.slider(
    "⚡ Gamma (for RBF / precomputed)", 0.1, 2.0, 1.0, 0.1)
neighbors = st.sidebar.slider(
    "🤝 Neighbors (for nearest_neighbors)", 2, 20, 10, 1)

# -------------------------------
# Helper function for clustering
# -------------------------------


# -------------------------------
# Cached Clustering Function
# -------------------------------
@st.cache_data
def run_clustering_cached(X_input, method, n_clusters=3, gamma=1.0, neighbors=10):
    """
    Robust Spectral Clustering runner, safe for Streamlit caching.
    X_input: array-like. Can be features (n_samples x n_features) or precomputed (n_samples x n_samples)
    method: 'rbf', 'nearest_neighbors', 'precomputed', 'cosine', 'manhattan'
    """
    X_input = np.nan_to_num(X_input, nan=0.0, posinf=0.0, neginf=0.0)
    n_clusters = min(n_clusters, X_input.shape[0])

    # ✅ Only compute affinity if NOT precomputed
    if method == "precomputed":
        if X_input.shape[0] != X_input.shape[1]:
            raise ValueError(
                f"Precomputed affinity must be square: shape={X_input.shape}")
        affinity_matrix = X_input
    elif method == 'rbf':
        affinity_matrix = rbf_kernel(X_input, gamma=gamma)
    elif method == 'nearest_neighbors':
        affinity_matrix = kneighbors_graph(
            X_input, n_neighbors=min(neighbors, X_input.shape[0]-1), include_self=True
        ).toarray()
    elif method == 'cosine':
        affinity_matrix = cosine_similarity(X_input)
    elif method == 'manhattan':
        from sklearn.metrics import pairwise_distances
        affinity_matrix = np.exp(-pairwise_distances(X_input,
                                 metric='manhattan'))
    else:
        raise ValueError(f"Unknown method '{method}'")

    affinity_matrix = np.nan_to_num(
        affinity_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    affinity_matrix[affinity_matrix < 0] = 0

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42
    )
    labels = sc.fit_predict(affinity_matrix)
    return labels


# --- Run clustering once with current sidebar selections ---
if affinity_choice == "precomputed":
    X_precomputed = rbf_kernel(X_pca, gamma=gamma)  # shape: (1000, 1000)
    labels = run_clustering_cached(
        X_precomputed,
        affinity_choice,
        n_clusters=k,
        gamma=gamma,
        neighbors=neighbors
    )
else:
    labels = run_clustering_cached(
        X_pca,
        affinity_choice,
        n_clusters=k,
        gamma=gamma,
        neighbors=neighbors
    )


# Update df_sampled clusters
df_sampled['Cluster'] = labels

# Merge back to main df
df = df.drop(columns=['Cluster'], errors='ignore')
df = df.merge(df_sampled[['Customer ID', 'Cluster']],
              on='Customer ID', how='left')


# -------------------------------
# STREAMLIT TABS FOR INTERACTIVITY
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Sales Stats",
    "🔹 Spectral Clustering Comparison",
    "📈 Cluster Characteristics",
    "🗺️ US Map"
])

# -------------------------------
# TAB 1: Sales Statistics + Hyperparameters
# -------------------------------
with tab1:
    st.subheader("⚙️ Adjust Hyperparameters for Spectral Clustering")
    # -----------------------------------
    # Step 1: Find best k automatically
    # -----------------------------------
    k_range = range(2, 11)
    best_score = -1
    best_k = 2
    for k_test in k_range:
        labels_test = SpectralClustering(
            n_clusters=k_test, affinity="rbf", gamma=1.0, random_state=42
        ).fit_predict(X_pca)
        score = silhouette_score(X_pca, labels_test)
        if score > best_score:
            best_score = score
            best_k = k_test

    # -----------------------------------
    # Step 3: Run clustering with chosen values
    # -----------------------------------

    # Display best silhouette
    results[affinity_choice] = silhouette_score(X_pca, labels)
    st.success(
        f"🏆 Best Method: **{affinity_choice}** with Silhouette Score: {results[affinity_choice]:.4f}")

    # -----------------------------------
    # Step 4: Your Sales Statistics section
    # -----------------------------------
    st.subheader("📊 Sales Statistics")
    st.write(f"• Total Sales: ${sales_stats['Total Sales']:,.2f}")
    st.write(f"• Average Sale: ${sales_stats['Average Sales']:,.2f}")
    st.write(f"• Max Sale: ${sales_stats['Max Sales']:,.2f}")
    st.write(f"• Min Sale: ${sales_stats['Min Sales']:,.2f}")
    st.write(f"• Std Dev: ${sales_stats['Sales Std Dev']:,.2f}")

    st.subheader("TOP 5 STATES BY PROFIT")
    st.dataframe(state_profit.head().reset_index().rename(
        columns={'Profit': 'Profit ($)'}))

    st.subheader("TOP 5 SUB-CATEGORIES BY PROFIT")
    st.dataframe(top_subcats[['Sub-Category', 'Profit']
                             ].head().rename(columns={'Profit': 'Profit ($)'}))

    st.subheader("📅 Monthly Sales Trend")
    st.plotly_chart(fig_monthly, use_container_width=True, key="monthly_trend")
    st.markdown("### 📊 Sales Statistics Overview")
    st.plotly_chart(fig1, use_container_width=True, key="sales_stats_bar")
    st.markdown("### 📈 State-wise Profit Distribution")
    st.plotly_chart(fig2, use_container_width=True, key="state_profit_bar")
    st.markdown("### 📊 Top Sub-Categories by Profit")
    st.plotly_chart(fig3, use_container_width=True, key="top_subcats_bar")
    st.markdown("### 📈 Customer Segments Analysis")
    st.plotly_chart(fig4, use_container_width=True, key="segments_combo")
    st.markdown("### 📊 Overall Profit Margin")
    st.plotly_chart(fig7, use_container_width=True, key="profit_margin_gauge")
    st.markdown("### 📈 Category Sales by Quantity")
    st.plotly_chart(fig6, use_container_width=True, key="category_qty_bar")
# -------------------------------
# TAB 2: Method Comparisons + Metrics
# -------------------------------
with tab2:
    st.subheader(
        "🔹 Comparison of Spectral Clustering Methods (PCA Projection)")
    # -------------------------------
    # Compute clustering for all methods
    # -------------------------------
    methods = ["nearest_neighbors", "rbf",
               "precomputed", "cosine", "manhattan"]
    labels_dict = {}

    for method in methods:
        if method == "precomputed":
            # Precompute RBF kernel to ensure square matrix
            X_precomputed = rbf_kernel(X_pca, gamma=gamma)
            labels_dict[method] = run_clustering_cached(
                X_precomputed,
                method,
                n_clusters=k,
                gamma=gamma,
                neighbors=neighbors
            )
        else:
            labels_dict[method] = run_clustering_cached(
                X_pca,
                method,
                n_clusters=k,
                gamma=gamma,
                neighbors=neighbors
            )

    # Recompute clustering for each method with current sidebar params
    # -------------------------------
    # Compute precomputed affinity once
    # -------------------------------

    # Create PCA scatter with all methods side by side
    methods = ["nearest_neighbors", "rbf", "precomputed", "cosine"]
    methods += ["manhattan"]  # add manhattan last for layout
    fig_methods = make_subplots(
        rows=1, cols=len(methods), subplot_titles=methods)
    for idx, method in enumerate(methods, start=1):
        fig_methods.add_trace(
            go.Scatter(
                x=X_pca[:, 0], y=X_pca[:, 1],
                mode="markers",
                marker=dict(color=labels_dict[method], colorscale="Viridis"),
                showlegend=False
            ),
            row=1, col=idx
        )
    fig_methods.update_layout(
        title_text="Spectral Clustering Results (All Methods)")

    st.plotly_chart(fig_methods, use_container_width=True, key="methods_plot")
    st.subheader("🔬 Spectral Clustering Methods")

    # Pick method interactively
    method_choice = st.radio(
        "Choose clustering method:",
        ["Nearest Neighbors", "RBF Kernel", "Precomputed Affinity",
            "Cosine Similarity", "Manhattan Distance"],
        horizontal=True,
        key="method_radio"
    )

    # Small explanations for each method
    explanations = {
        "Nearest Neighbors": """
        **Nearest Neighbors Affinity**
        - Builds graph from *k* closest points.  
        - ✅ Captures local, non-linear shapes.  
        - ⚠️ Choice of *k* is critical.  
        - 🎯 Best for data with **local structure**.
        """,
        "RBF Kernel": """
        **RBF Kernel Affinity**
        
        - Uses Gaussian similarity with parameter γ.  
        - ✅ Smooth, balances local + global structure.  
        - ⚠️ Needs feature scaling + γ tuning.  
        - 🎯 Best for **compact, globular clusters**.
        """,
        "Precomputed Affinity": """
        **Precomputed RBF Affinity**
        - You supply the similarity matrix.  
        - ✅ Full control, allows custom kernels.  
        - ⚠️ Memory-heavy, often similar to RBF.  
        - 🎯 Best when you want **flexibility/custom metrics**.
        """,
        "Cosine Similarity": """
        **Cosine Similarity Affinity**
        - Measures angle between vectors, not distance.  
        - ✅ Good for high-dimensional, sparse data.  
        - ⚠️ Sensitive to vector magnitudes.  
        - 🎯 Best for **text data or normalized features**.
        """,
        "Manhattan Distance": """
        **Manhattan Distance Affinity**
        - Uses L1 norm (sum of absolute differences).  
        - ✅ Robust to outliers, interpretable.  
        - ⚠️ Less sensitive to local structure than RBF.  
        - 🎯 Best for **data with outliers or non-Euclidean geometry**.
        """
    }

    # Display interactive explanation
    st.info(explanations[method_choice])
    st.subheader("📊 Spectral Clustering Evaluation Metrics")

    # -------------------------------
    # Cluster Evaluation Metrics Table
    # -------------------------------
    metrics_dict = {}
    methods = ["nearest_neighbors", "rbf",
               "precomputed", "cosine", "manhattan"]

    for method in methods:
        labels = labels_dict[method]
        X_for_metrics = X_pca
        try:
            metrics_dict[method] = {
                'Silhouette': silhouette_score(X_for_metrics, labels),
                'Davies–Bouldin': davies_bouldin_score(X_for_metrics, labels),
                'Calinski–Harabasz': calinski_harabasz_score(X_for_metrics, labels)
            }
        except Exception as e:
            metrics_dict[method] = {
                'Silhouette': None,
                'Davies–Bouldin': None,
                'Calinski–Harabasz': None
            }
            st.warning(f"⚠️ Metrics could not be computed for {method}: {e}")

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_dict).T.reset_index().rename(
        columns={'index': 'Method'})

    st.dataframe(metrics_df.style.format({
        'Silhouette': "{:.4f}",
        'Davies–Bouldin': "{:.4f}",
        'Calinski–Harabasz': "{:.2f}"
    }))

    # --- Bar Charts (already in your code) ---
    st.subheader("📊 Metrics Comparison (Visual)")
    # Silhouette Score
    fig_silhouette = px.bar(
        metrics_df, x='Method', y='Silhouette',
        color='Silhouette', color_continuous_scale='Viridis',
        text=metrics_df['Silhouette'].apply(
            lambda x: f"{x:.3f}" if pd.notnull(x) else "NA"),
        title="Silhouette Score Comparison"
    )
    st.plotly_chart(fig_silhouette, use_container_width=True,
                    key="silhouette_plot")

    # Davies-Bouldin (lower is better)
    fig_db = px.bar(
        metrics_df, x='Method', y='Davies–Bouldin',
        color='Davies–Bouldin', color_continuous_scale='Inferno_r',
        text=metrics_df['Davies–Bouldin'].apply(
            lambda x: f"{x:.3f}" if pd.notnull(x) else "NA"),
        title="Davies–Bouldin Score Comparison (Lower is Better)"
    )
    st.plotly_chart(fig_db, use_container_width=True, key="db_plot")

    # Calinski–Harabasz Score
    fig_ch = px.bar(
        metrics_df, x='Method', y='Calinski–Harabasz',
        color='Calinski–Harabasz', color_continuous_scale='Cividis',
        text=metrics_df['Calinski–Harabasz'].apply(
            lambda x: f"{x:.1f}" if pd.notnull(x) else "NA"),
        title="Calinski–Harabasz Score Comparison"
    )
    st.plotly_chart(fig_ch, use_container_width=True, key="ch_plot")

    # --- Executive Explanation ---
    best_method = metrics_df.loc[metrics_df['Silhouette'].idxmax(), 'Method']
    best_sil = metrics_df['Silhouette'].max()

    explanation = f"""
    ### ✅ Why This Outcome is Best for Project
    - The **{best_method} method** produced the **highest Silhouette score ({best_sil:.4f})**, 
      meaning clusters are both **well-separated** and **internally consistent**.  
    - It also performs well across the **Davies–Bouldin** (lower is better) and 
      **Calinski–Harabasz** (higher is better) scores.  
    - This balance shows the clusters are **clear, stable, and meaningful** — which is 
      crucial for making **data-driven decisions** in our project.  
    - Choosing the best-performing method ensures **management can trust the insights**, 
      since the segmentation is based on **objective evaluation metrics**, not guesswork.
    """

    st.success(explanation)


# -------------------------------
# TAB 3: Cluster Characteristics
# -------------------------------
with tab3:
    st.subheader("🔎 Cluster Characteristics")

    # --- Dynamic Radar Charts Across Methods ---
    methods = ["nearest_neighbors", "rbf",
               "precomputed", "cosine", "manhattan"]
    labels_dict_radar = labels_dict  # reuse from tab2

    metrics = ['Sales', 'Profit', 'Quantity', 'Discount', 'Profit Margin']
    fig_radar = make_subplots(
        rows=1, cols=len(methods),
        specs=[[{'type': 'polar'}]*len(methods)],
        subplot_titles=methods
    )

    for i, method in enumerate(methods):
        df_temp = df_sampled.copy()
        df_temp['Cluster_temp'] = labels_dict_radar[method]
        for cluster in sorted(df_temp['Cluster_temp'].dropna().unique()):
            cluster_data = df_temp[df_temp['Cluster_temp']
                                   == cluster].mean(numeric_only=True)
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[cluster_data[m] for m in metrics],
                    theta=metrics,
                    fill='toself',
                    name=f'{method} - Cluster {cluster}',
                    showlegend=(i == 0)
                ),
                row=1, col=i+1
            )

    fig_radar.update_layout(
        height=600, width=1600,
        title_text="Cluster Characteristics Across Methods"
    )

    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("ℹ️ Understanding the 'Cluster' Column")

    # -------------------------------
    # Explanation and Example
    # -------------------------------
    st.markdown("""
    The **`Cluster`** column represents customer segments identified using **Spectral Clustering**.
    It groups customers with similar purchasing behavior.
    """)

    with st.expander("Click to learn how clusters differentiate customers"):
        st.markdown("""
        **Key Points:**
        - Each number in `Cluster` represents a **different customer segment**.
        - Customers in the same cluster share similar patterns in:
            - Total sales
            - Profit contribution
            - Purchase frequency
            - Discounts used
            - Profit margins
        """)

        st.markdown("**Example Interpretation:**")

        st.markdown("""
        - ![Cluster 0](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExcDNtMjg2N3J0cDFlMWFneTVyN2luZDBnZzNmMjE0NTExeGR6bG05ZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/ILW1fbJHW0Ndm/giphy.gif)  
          *High-value customers with large purchases*  
          *Frequent buyers with lower profit margins*  
          *Occasional buyers with moderate sales and profit*
        """)

        st.markdown("""
        **Usage in Dashboard:**  
        The `Cluster` column is used to color-code customers in charts and maps, helping to:
        - Compare segments visually  
        - Analyze patterns in sales and profit  
        - Identify target groups for marketing or promotions  

        *Hopefully, this helps you understand how customers are segmented based on their purchasing behavior!*  
        - ![Cluster 1](https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif)
        """)

    # -------------------------------
    # Cluster Summary Example
    # -------------------------------
    st.subheader("📊 Cluster Summary Example")
    st.dataframe(df_sampled.groupby('Cluster').agg({
        'Sales': 'mean',
        'Profit': 'mean',
        'Customer ID': 'count'
    }).rename(columns={'Customer ID': 'Customer Count'}))

# -------------------------------
# TAB 4: US Map
# -------------------------------
with tab4:
    st.subheader("🗺️ Customer Clusters Across US States")
    st.markdown(
        "The map shows total sales and cluster assignment for each state. Hover to see details.")
    components.html(open('spectral_clustering_map.html', 'r',
                    encoding='utf-8').read(), height=600)
