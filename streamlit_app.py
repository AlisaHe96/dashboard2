import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger('streamlit.runtime.scriptrunner_utils').setLevel(logging.ERROR)
logging.getLogger('streamlit').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import plotly.express as px
import matplotlib.pyplot as plt

# Page and Theme Settings
st.set_page_config(
    page_title="Cognitive Performance Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown('''
    <style>
        /* Hide Streamlit Default Menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        /* Card Shadow */
        .stMetric {box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 8px;}
    </style>
''', unsafe_allow_html=True)

# Sidebar Information
with st.sidebar:
    st.image("brain_icon.png", use_container_width=True)
    
    st.title("🧠 Dashboard Explanation")
    st.write("Predict and Cluster Cognitive Performance. Select different tasks on the left to view results.")
    st.markdown("---")
    st.write("**Data Source:** human_cognitive_performance.csv")
    st.write("**Processing:** Standardization, SMOTE, One-Hot Encoding")
    st.markdown("---")
    st.write("© Ziqi (Michael) Wang, Mingyue Tang , Jiahong Wen, Yu Xue, Henry Zhou, Alisa He")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("human_cognitive_performance.csv")
        
        if 'Cognitive_Score' not in df.columns:
            st.error("Error: 'Cognitive_Score' column not found in the CSV file.")
            st.stop()

        df['Cognitive_Level'] = pd.cut(
            df['Cognitive_Score'],
            bins=[-np.inf, 40, 70, np.inf],
            labels=['Low', 'Medium', 'High']
        )
        
        df.drop(columns=['User_ID', 'AI_Predicted_Score'], inplace=True, errors='ignore')
        
        numeric_cols_orig = ['Age', 'Sleep_Duration', 'Stress_Level', 'Daily_Screen_Time', 'Caffeine_Intake', 'Reaction_Time', 'Memory_Test_Score']
        categorical_cols_orig = ['Gender', 'Diet_Type', 'Exercise_Frequency']

        # Check if all numeric_cols are present
        missing_numeric_cols = [col for col in numeric_cols_orig if col not in df.columns]
        if missing_numeric_cols:
            st.error(f"Error: Missing essential numeric columns: {', '.join(missing_numeric_cols)}")
            st.stop()

        # Create df_eda (copy of df before scaling and get_dummies)
        # This df_eda will have Cognitive_Level, original numeric, and original categorical columns
        df_eda = df.copy()
        
        # Apply StandardScaler to df (for ML)
        scaler = StandardScaler()
        df[numeric_cols_orig] = scaler.fit_transform(df[numeric_cols_orig])
        
        # Only apply one-hot encoding to existing categorical columns for df (for ML)
        existing_categorical_cols_for_dummies = [col for col in categorical_cols_orig if col in df.columns]
        df = pd.get_dummies(df, columns=existing_categorical_cols_for_dummies)
        
        return df, numeric_cols_orig, df_eda, categorical_cols_orig

    except FileNotFoundError:
        st.error("Error: 'human_cognitive_performance.csv' not found.")
        st.info("Please ensure the CSV file is in the same directory as the Streamlit script.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        st.stop()

# Load Data
# df_processed_ml: DataFrame for ML models (scaled, one-hot encoded)
# numeric_cols_for_scaling: Names of original numeric columns used for scaling
# df_eda: DataFrame for EDA (original values, no one-hot encoding on categoricals)
# original_categorical_cols: Names of original categorical columns
df_processed_ml, numeric_cols_for_scaling, df_eda, original_categorical_cols = load_data()

# Data Check Hint: Display data summary in sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("📊 Data Summary")
    st.write(f"**Rows:** {df_processed_ml.shape[0]}")
    st.write(f"**Columns:** {df_processed_ml.shape[1]}")
    st.markdown("---")
    st.subheader("Initial Data Preview (first 5 rows)")
    st.dataframe(df_processed_ml.head())
    st.markdown("---")

# Task Tabs - Added EDA tab
tab_overview, tab_reg, tab_clf, tab_clu, tab_eda = st.tabs(["📊 Overview", "🔢 Regression", "🎯 Classification", "🔍 Clustering", "📈 EDA"])

# Overview Tab
with tab_overview:
    st.header("📊 Dashboard Overview")
    
    # FIX: Grouped Average Score and Sample Count in one column
    metric_col, chart_col = st.columns([1, 2]) # Adjust column ratios as needed

    with metric_col:
        avg_score = df_eda['Cognitive_Score'].mean()
        sample_count = df_processed_ml.shape[0] 

        st.metric("Average Score", f"{avg_score:.2f}")
        st.metric("Sample Count", f"{sample_count}") 

    with chart_col:
        dist = df_eda['Cognitive_Level'].value_counts(normalize=True) * 100
        st.subheader("Cognitive Level Distribution (%)")
        st.bar_chart(dist)
        st.write("This chart shows the proportion of individuals categorized into Low, Medium, and High cognitive levels.")

    st.markdown("---")
    st.subheader("Key Statistics")
    # Use df_eda for descriptive statistics as it has original values
    st.dataframe(df_eda[numeric_cols_for_scaling + ['Cognitive_Score']].describe().transpose())


# Regression Tab
with tab_reg:
    st.header("🔢 Regression: Predict Cognitive Score")
    
    # Prepare data (using df_processed_ml)
    X_cols = [*numeric_cols_for_scaling, *[c for c in df_processed_ml if c.startswith(('Gender_','Diet_Type_','Exercise_Frequency_'))]]
    X_cols_existing = [col for col in X_cols if col in df_processed_ml.columns]
    X = df_processed_ml[X_cols_existing]
    y = df_processed_ml['Cognitive_Score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.subheader("Model Selection and Hyperparameters")
    selected_model_reg = st.selectbox("Select Regression Model", ["Linear Regression", "Random Forest", "Neural Network", "SVR"], key="reg_model_select_top")

    model = None # Initialize model to None

    # Conditional Hyperparameter Display
    if selected_model_reg == "Linear Regression":
        st.info("Linear Regression typically does not require hyperparameter tuning.")
        model = LinearRegression()
    elif selected_model_reg == "Random Forest":
        n_estimators = st.slider("Random Forest: n_estimators", 10, 500, 100, key="reg_rf_estimators")
        max_depth = st.slider("Random Forest: max_depth (0 for no limit)", 0, 50, 10, key="reg_rf_max_depth")
        min_samples_leaf = st.slider("Random Forest: min_samples_leaf", 1, 20, 1, key="reg_rf_min_samples_leaf")
        
        # Convert 0 to None for max_depth
        actual_max_depth = None if max_depth == 0 else max_depth
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=actual_max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    elif selected_model_reg == "Neural Network":
        hidden_layer1_size = st.slider("Neural Network: Hidden Layer 1 Size", 50, 500, 100, key="reg_nn_hidden1")
        add_second_layer = st.checkbox("Add Second Hidden Layer?", key="reg_nn_add_second")
        
        hidden_layer_sizes_tuple = (hidden_layer1_size,)
        if add_second_layer:
            hidden_layer2_size = st.slider("Neural Network: Hidden Layer 2 Size", 20, 200, 50, key="reg_nn_hidden2")
            hidden_layer_sizes_tuple = (hidden_layer1_size, hidden_layer2_size)

        activation = st.selectbox("Neural Network: Activation Function", ['relu', 'tanh', 'logistic'], key="reg_nn_activation")
        solver = st.selectbox("Neural Network: Solver", ['adam', 'sgd', 'lbfgs'], key="reg_nn_solver")
        alpha = st.slider("Neural Network: Alpha (L2 penalty)", 0.0001, 0.1, 0.0001, format="%.4f", key="reg_nn_alpha")

        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes_tuple,
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=500, # Kept fixed for simplicity
            random_state=42
        )
    elif selected_model_reg == "SVR":
        C = st.slider("SVR: C (Regularization parameter)", 0.01, 10.0, 1.0, key="reg_svr_c")
        kernel = st.selectbox("SVR: Kernel Type", ['rbf', 'linear', 'poly', 'sigmoid'], key="reg_svr_kernel")
        
        degree = 3 # Default for non-poly kernels
        if kernel == 'poly':
            degree = st.slider("SVR: Degree (for 'poly' kernel)", 1, 5, 3, key="reg_svr_degree")
            
        gamma_option = st.selectbox("SVR: Gamma (Kernel coefficient)", ['scale', 'auto', 'custom'], key="reg_svr_gamma_option")
        gamma_val = gamma_option
        if gamma_option == 'custom':
            gamma_val = st.slider("SVR: Custom Gamma Value", 0.001, 1.0, 0.1, format="%.3f", key="reg_svr_gamma_custom")
            
        epsilon = st.slider("SVR: Epsilon (Insensitivity zone)", 0.01, 1.0, 0.1, key="reg_svr_epsilon")

        model = SVR(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma_val,
            epsilon=epsilon
        )
    
    if model is not None: # Ensure a model is selected before showing the train button
        st.markdown("---")
        if st.button("Train Regression Model", key="train_reg_button"):
            with st.spinner(f"Training {selected_model_reg} Model..."):
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.success(f"Test Mean Squared Error (MSE): {mse:.3f}")
            fig = px.scatter(x=y_test, y=y_pred, labels={'x':'Actual Score','y':'Predicted Score'}, title="Actual vs Predicted Cognitive Score")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select a model to configure its hyperparameters and train.")


# Classification Tab
with tab_clf:
    st.header("🎯 Classification: Predict Cognitive Level")
    
    # Prepare data (using df_processed_ml)
    X_cols = [*numeric_cols_for_scaling, *[c for c in df_processed_ml if c.startswith(('Gender_','Diet_Type_','Exercise_Frequency_'))]]
    X_cols_existing = [col for col in X_cols if col in df_processed_ml.columns]
    X = df_processed_ml[X_cols_existing]
    y = df_processed_ml['Cognitive_Level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with st.spinner("Applying SMOTE for class balancing..."):
        X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
    st.success("SMOTE application complete.")

    st.subheader("Model Selection and Hyperparameters")
    selected_model_clf = st.selectbox("Select Classification Model", ["Random Forest", "SVM", "Logistic Regression", "Neural Network"], key="clf_model_select_top")

    model = None # Initialize model to None

    # Conditional Hyperparameter Display
    if selected_model_clf == "Random Forest":
        n_estimators = st.slider("Random Forest: n_estimators", 10, 500, 100, key="clf_rf_estimators")
        max_depth = st.slider("Random Forest: max_depth (0 for no limit)", 0, 50, 10, key="clf_rf_max_depth")
        min_samples_leaf = st.slider("Random Forest: min_samples_leaf", 1, 20, 1, key="clf_rf_min_samples_leaf")
        criterion = st.selectbox("Random Forest: Criterion", ['gini', 'entropy', 'log_loss'], key="clf_rf_criterion")
        
        actual_max_depth = None if max_depth == 0 else max_depth
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=actual_max_depth,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42
        )
    elif selected_model_clf == "SVM":
        C = st.slider("SVM: C (Regularization parameter)", 0.01, 10.0, 1.0, key="clf_svm_c")
        kernel = st.selectbox("SVM: Kernel Type", ['rbf', 'linear', 'poly', 'sigmoid'], key="clf_svm_kernel")
        
        degree = 3 # Default for non-poly kernels
        if kernel == 'poly':
            degree = st.slider("SVM: Degree (for 'poly' kernel)", 1, 5, 3, key="clf_svm_degree")
            
        gamma_option = st.selectbox("SVM: Gamma (Kernel coefficient)", ['scale', 'auto', 'custom'], key="clf_svm_gamma_option")
        gamma_val = gamma_option
        if gamma_option == 'custom':
            gamma_val = st.slider("SVM: Custom Gamma Value", 0.001, 1.0, 0.1, format="%.3f", key="clf_svm_gamma_custom")
            
        model = SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma_val,
            probability=True # Kept True as per original
        )
    elif selected_model_clf == "Logistic Regression":
        C = st.slider("Logistic Regression: C (Inverse of regularization strength)", 0.01, 10.0, 1.0, key="clf_lr_c")
        solver = st.selectbox("Logistic Regression: Solver", ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'], key="clf_lr_solver")
        
        # Filter penalty options based on solver
        penalty_options = []
        if solver in ['lbfgs', 'newton-cg', 'sag']:
            penalty_options.append('l2')
        if solver == 'liblinear':
            penalty_options.extend(['l1', 'l2'])
        if solver == 'saga':
            penalty_options.extend(['l1', 'l2', 'elasticnet'])
        
        # Ensure penalty_options is not empty, provide a default if it would be
        if not penalty_options:
            penalty_options = ['none'] # Or handle this case as an error or specific message
        
        penalty = st.selectbox("Logistic Regression: Penalty", penalty_options, key="clf_lr_penalty")

        l1_ratio = None
        if penalty == 'elasticnet':
            l1_ratio = st.slider("Logistic Regression: L1 Ratio (for elasticnet)", 0.0, 1.0, 0.5, key="clf_lr_l1_ratio")

        model = LogisticRegression(
            C=C,
            solver=solver,
            penalty=penalty,
            l1_ratio=l1_ratio,
            max_iter=500 # Kept fixed for simplicity
        )
    elif selected_model_clf == "Neural Network":
        hidden_layer1_size = st.slider("Neural Network: Hidden Layer 1 Size", 50, 500, 100, key="clf_nn_hidden1")
        add_second_layer = st.checkbox("Add Second Hidden Layer?", key="clf_nn_add_second")
        
        hidden_layer_sizes_tuple = (hidden_layer1_size,)
        if add_second_layer:
            hidden_layer2_size = st.slider("Neural Network: Hidden Layer 2 Size", 20, 200, 50, key="clf_nn_hidden2")
            hidden_layer_sizes_tuple = (hidden_layer1_size, hidden_layer2_size)

        activation = st.selectbox("Neural Network: Activation Function", ['relu', 'tanh', 'logistic'], key="clf_nn_activation")
        solver = st.selectbox("Neural Network: Solver", ['adam', 'sgd', 'lbfgs'], key="clf_nn_solver")
        alpha = st.slider("Neural Network: Alpha (L2 penalty)", 0.0001, 0.1, 0.0001, format="%.4f", key="clf_nn_alpha")

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes_tuple,
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=500, # Kept fixed for simplicity
            random_state=42
        )
    
    if model is not None: # Ensure a model is selected before showing the train button
        st.markdown("---")
        if st.button("Train Classification Model", key="train_clf_button"):
            with st.spinner(f"Training {selected_model_clf} Model..."):
                model.fit(X_res, y_res)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Accuracy: {acc:.2%}")
            
            cm = confusion_matrix(y_test, y_pred, labels=['Low','Medium','High'])
            fig, ax = plt.subplots()
            im = ax.imshow(cm, cmap='Blues') 
            
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha='center', va='center', color='black' if cm[i,j] < im.norm(im.get_array().max())/2 else 'white') 
                    
            ax.set_xticks(range(3)); ax.set_xticklabels(['Low','Medium','High'])
            ax.set_yticks(range(3)); ax.set_yticklabels(['Low','Medium','High'])
            ax.set_xlabel('Predicted Cognitive Level'); ax.set_ylabel('True Cognitive Level')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
    else:
        st.warning("Please select a model to configure its hyperparameters and train.")


# Clustering Tab
with tab_clu:
    st.header("🔍 Clustering: Unsupervised Learning")
    
    X = df_processed_ml[numeric_cols_for_scaling] # Only use numeric features for clustering (scaled ones for KMeans, GMM default)
    
    st.subheader("Model Selection and Hyperparameters")
    selected_model_clu = st.selectbox("Select Clustering Model", ["KMeans", "Hierarchical", "GMM"], key="clu_model_select_top")

    model = None # Initialize model to None

    # Conditional Hyperparameter Display
    if selected_model_clu == "KMeans":
        n_clusters = st.slider("KMeans: number of clusters", 2, 10, 3, key="clu_kmeans_clusters")
        init_method = st.selectbox("KMeans: Initialization Method", ['k-means++', 'random'], key="clu_kmeans_init")
        max_iter = st.slider("KMeans: Max Iterations", 100, 1000, 300, key="clu_kmeans_max_iter")
        
        model = KMeans(
            n_clusters=n_clusters,
            init=init_method,
            max_iter=max_iter,
            random_state=42,
            n_init='auto'
        )
    elif selected_model_clu == "Hierarchical":
        n_clusters = st.slider("Hierarchical: number of clusters", 2, 10, 3, key="clu_agg_clusters")
        linkage_type = st.selectbox("Hierarchical: Linkage Type", ['ward', 'complete', 'average', 'single'], key="clu_agg_linkage")
        
        affinity_options = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
        if linkage_type == 'ward':
            affinity_options = ['euclidean'] # 'ward' linkage only supports 'euclidean' metric.
        affinity_type = st.selectbox("Hierarchical: Affinity Metric", affinity_options, key="clu_agg_affinity")
        
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_type,
            metric=affinity_type 
        )
    elif selected_model_clu == "GMM":
        n_components = st.slider("GMM: number of components", 2, 10, 3, key="clu_gmm_components")
        covariance_type = st.selectbox("GMM: Covariance Type", ['full', 'tied', 'diag', 'spherical'], key="clu_gmm_covariance")
        max_iter = st.slider("GMM: Max EM Iterations", 50, 1000, 100, key="clu_gmm_max_iter")
        
        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=42
        )
    
    if model is not None: # Ensure a model is selected before showing the run button
        st.markdown("---")
        if st.button("Run Clustering Algorithm", key="run_clu_button"):
            with st.spinner(f"Running {selected_model_clu} Algorithm..."):
                labels = model.fit_predict(X)
                
            df_processed_ml['Cluster'] = labels.astype(str) # Add cluster labels to the processed df for visualization
            st.success("Clustering completed and clusters are visualized.")
            
            # Display cluster sizes as a hint
            st.subheader("Cluster Sizes:")
            st.write(df_processed_ml['Cluster'].value_counts().sort_index())

            fig = px.scatter(df_processed_ml, x='Age', y='Cognitive_Score', color='Cluster', 
                             title='Cluster Visualization (Age vs. Cognitive Score)',
                             hover_data=numeric_cols_for_scaling) # Add hover data for more insights
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select a model to configure its hyperparameters and run.")

# EDA Tab
with tab_eda:
    st.header("📈 Exploratory Data Analysis (EDA)")
    st.markdown("---")

    st.subheader("Correlation Heatmap of Numerical Features")
    st.write("This heatmap shows the Pearson correlation coefficient between pairs of numerical features. Values close to 1 or -1 indicate a strong positive or negative linear relationship, respectively. Values close to 0 indicate a weak linear relationship.")
    
    # Select only numerical columns from df_eda for correlation
    correlation_df = df_eda[numeric_cols_for_scaling + ['Cognitive_Score']]
    correlation_matrix = correlation_df.corr()
    fig_heatmap = px.imshow(correlation_matrix, text_auto=True, aspect="auto", 
                            color_continuous_scale='RdBu_r', range_color=[-1,1],
                            title="Correlation Heatmap of Numerical Features")
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown("---")

    st.subheader("Individual Feature Distributions")
    st.write("Explore the distribution of individual numerical features using histograms and box plots.")
    selected_eda_numeric_col = st.selectbox("Select Numeric Feature for Distribution Plot", 
                                            numeric_cols_for_scaling + ['Cognitive_Score'], 
                                            key="eda_dist_select")
    fig_hist = px.histogram(df_eda, x=selected_eda_numeric_col, marginal="box", 
                            title=f"Distribution of {selected_eda_numeric_col}",
                            color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("---")

    st.subheader("Categorical Feature Distributions")
    st.write("View the distribution of various categorical features.")
    for col in original_categorical_cols:
        fig_cat = px.bar(df_eda, x=col, title=f"Distribution of {col}", 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_cat, use_container_width=True)
    st.markdown("---")

    st.subheader("Grouped Box Plots: Numerical Features by Category")
    st.write("Examine how numerical features vary across different categories (e.g., Cognitive Score by Diet Type).")
    selected_eda_num_col_grouped = st.selectbox("Select Numeric Feature for Grouped Plot", 
                                                 numeric_cols_for_scaling + ['Cognitive_Score'], 
                                                 key="eda_grouped_num_select")
    selected_eda_cat_col_grouped = st.selectbox("Select Categorical Feature for Grouped Plot", 
                                                 original_categorical_cols + ['Cognitive_Level'], 
                                                 key="eda_grouped_cat_select")
    fig_box = px.box(df_eda, x=selected_eda_cat_col_grouped, y=selected_eda_num_col_grouped, 
                     title=f"{selected_eda_num_col_grouped} by {selected_eda_cat_col_grouped}",
                     color=selected_eda_cat_col_grouped,
                     color_discrete_sequence=px.colors.qualitative.Dark24) # Use color
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown("---")

    st.subheader("Scatter Plots: Relationships Between Numerical Features")
    st.write("Visualize the relationship between two numerical features, with an option to color points by a categorical variable.")
    col_scatter1, col_scatter2 = st.columns(2)
    with col_scatter1:
        x_scatter_col = st.selectbox("Select X-axis Feature", 
                                     numeric_cols_for_scaling + ['Cognitive_Score'], 
                                     key="eda_scatter_x")
    with col_scatter2:
        y_scatter_col = st.selectbox("Select Y-axis Feature", 
                                     numeric_cols_for_scaling + ['Cognitive_Score'], 
                                     key="eda_scatter_y")
    
    color_scatter_col = st.selectbox("Color points by (Optional)", 
                                     [None] + original_categorical_cols + ['Cognitive_Level'], 
                                     key="eda_scatter_color")
    
    fig_scatter = px.scatter(df_eda, x=x_scatter_col, y=y_scatter_col, color=color_scatter_col, 
                             title=f"Scatter Plot of {x_scatter_col} vs {y_scatter_col}",
                             hover_data=numeric_cols_for_scaling + original_categorical_cols + ['Cognitive_Score', 'Cognitive_Level'])
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("---")
