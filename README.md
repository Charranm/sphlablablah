© 2025 CHARRAN M. All rights reserved. 
This code may not be used, copied, or distributed without explicit permission.
Absolutely! Here's a polished **README.md** for your GitHub repository. It’s structured to explain the project clearly, highlight the dataset, spectral clustering methods, interactive dashboard, and usage instructions.

````markdown
# US E-Commerce Spectral Clustering Dashboard

## 📌 Project Overview

This project explores the **effectiveness of different spectral clustering algorithms** on a real-world dataset. The goal is to **analyze customer segmentation** for the 2020 US E-Commerce dataset and provide an **interactive, browser-based visualization tool** for insights.  

The dashboard allows users to **experiment with hyperparameters**, compare clustering methods, and visualize cluster characteristics geographically and analytically.

---

## 📊 Dataset

- **Dataset used:** https://www.kaggle.com/datasets/ammaraahmad/us-ecommerce-record-2020
- **Features included:** Sales, Quantity, Discount, Profit, Category, Segment, State, Customer ID, and more.  
- **Sample size for clustering:** 1,000 US customers (random sample for PCA and clustering visualization).  

> Note: You can substitute this dataset with any other dataset suitable for spectral clustering (e.g., [USA Flights Dataset](https://www.kaggle.com/flashgordon/usa-airport-dataset)).

---

## 🧩 Spectral Clustering Methods Implemented

1. **Nearest Neighbors Affinity**
   - Builds a graph from k nearest points.
   - Captures **local structure** in the data.
2. **RBF Kernel Affinity**
   - Uses Gaussian similarity with parameter γ.
   - Balances **local and global structure**.
3. **Precomputed Affinity (RBF Kernel)**
   - Full control via a precomputed similarity matrix.
   - Suitable for **custom metrics or kernels**.

**Hyperparameters included:**  
- Number of clusters (`k`)  
- Gamma (`γ`) for RBF  
- Number of neighbors (`n_neighbors`) for nearest neighbor affinity  

---

## 📈 Dashboard Features

The project is built in **Streamlit** and includes:

### Tab 1: Sales Statistics + Hyperparameters
- Interactive **sliders** for `k`, `gamma`, `neighbors`.
- Overview of total sales, average sale, top states by profit, top sub-categories.
- Monthly sales trend, category analysis, profit margin indicator.

### Tab 2: Spectral Clustering Comparison
- PCA scatter plots comparing all three spectral clustering methods.
- Evaluation metrics table: Silhouette, Davies–Bouldin, Calinski–Harabasz.
- **Interactive method selection** with concise explanation of each method.
- Highlights the **best clustering outcome** with justification for management.

### Tab 3: Cluster Characteristics
- Radar charts comparing clusters across metrics (Sales, Profit, Quantity, Discount, Profit Margin).
- Summary of cluster statistics for quick insight.

### Tab 4: US Map
- Interactive Folium map showing **customer clusters by state**.
- Hover popups for sales, profit, and cluster assignment.

---

## ⚙️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/Charranm/sphlablablah.git
````

2. Navigate to the folder:

```bash
cd us-ecommerce-spectral-clustering
```

3. Install required libraries:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit dashboard:

```bash
streamlit run app.py
```

> Ensure the dataset CSV file path matches your local environment or update the path in `app.py`.

---

## 📌 Key Benefits

* Demonstrates **effectiveness of different spectral clustering methods** on real data.
* Provides **interactive hyperparameter tuning** for experimentation.
* Generates **visual insights** suitable for management and decision-making.
* Flexible — dataset can be swapped for other use cases (e.g., flights, retail, customer behavior).

---

## 🛠️ Technologies Used

* Python 3.10+
* Streamlit (interactive dashboard)
* Pandas & NumPy (data processing)
* Scikit-learn (spectral clustering, PCA, metrics)
* Plotly (charts, radar plots)
* Folium (interactive US map)

---

## 📄 License

This project is for **educational/research purposes**. Please credit the author if used.

---

## 👨‍💻 Author

**CHARRAN MOHANRAJ**

* University of Liverpool – Applied AI Project
* GitHub: [yourgithubusername](https://github.com/Charranm)

