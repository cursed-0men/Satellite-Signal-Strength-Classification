import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Satellite Signal Strength Classification", 
    layout="wide", 
    page_icon="🛰️",
    initial_sidebar_state="collapsed"
)

# Set modern color palette
colors = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'warning': '#d62728',
    'info': '#17becf',
    'dark': '#2c3e50',
    'light': '#ecf0f1',
    'gradient_start': '#667eea',
    'gradient_end': '#764ba2'
}

# --------------------------
# Enhanced CSS Styling
# --------------------------
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {{
        background: linear-gradient(135deg, {colors['gradient_start']} 0%, {colors['gradient_end']} 100%);
        min-height: 100vh;
        padding: 2rem;
        font-family: 'Inter', sans-serif;
    }}
    
    .stApp {{
        background: transparent;
    }}
    
    /* Header Styling */
    .main-header {{
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        text-align: center;
    }}
    
    .main-title {{
        background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
        letter-spacing: -0.02em;
    }}
    
    .subtitle {{
        color: {colors['dark']};
        font-size: 1.2rem;
        font-weight: 400;
        margin-top: 0.5rem;
        opacity: 0.8;
    }}
    
    /* Card Styling */
    .metric-card {{
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, {colors['primary']}, {colors['secondary']});
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {colors['primary']};
        margin: 0;
        line-height: 1;
    }}
    
    .metric-label {{
        font-size: 0.9rem;
        color: {colors['dark']};
        font-weight: 500;
        margin-top: 0.5rem;
        opacity: 0.8;
    }}
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 4px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }}
    
    /* Input Styling */
    .input-section {{
        background: rgba(255, 76, 75, 0.9);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.3);
    }}
    
    .section-title {{
        color: {colors['dark']};
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .prediction-box {{
        background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']});
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        font-weight: 500;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }}
    
    /* Chart Container */
    .chart-container {{
        background: rgba(15, 17,22, 0.9);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }}
    
    /* Remove Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: rgba(255,255,255,0.3);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: rgba(255,255,255,0.5);
    }}
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Load Dataset
# --------------------------
@st.cache_data
def load_data():
    # For demo purposes, create sample data if CSV doesn't exist
    try:
        df = pd.read_csv("dataset/satellite_signal_dataset.csv")
    except:
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'rain_attenuation_db': np.random.normal(2.5, 1.2, n_samples),
            'cn_ratio_db': np.random.normal(15, 3, n_samples),
            'distance_m': np.random.uniform(1000, 50000, n_samples),
            'frequency_hz': np.random.uniform(2.4e9, 24e9, n_samples),
            'orbit_type': np.random.choice(['LEO', 'MEO', 'GEO'], n_samples),
            'signal_quality': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], n_samples)
        })
    
    labels, uniques = pd.factorize(df['signal_quality'])
    y = labels
    X_df = pd.get_dummies(df.drop(columns=['signal_quality']), drop_first=True)
    X = X_df.values.astype(np.float64)
    return df, X, X_df.columns, y, uniques

df, X, feature_columns, y, uniques = load_data()

# --------------------------
# Train-Test Split
# --------------------------
np.random.seed(42)
idx = np.arange(len(X))
np.random.shuffle(idx)
split = int(0.8 * len(X))
train_idx, test_idx = idx[:split], idx[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# --------------------------
# Enhanced Models with Progress Tracking
# --------------------------
class KNN:
    def __init__(self, k=5): 
        self.k = k
        self.name = "K-Nearest Neighbors"
    
    def fit(self, X, y): 
        self.X_train, self.y_train = X.astype(np.float64), y
    
    def predict_one(self, x):
        dist = np.linalg.norm(self.X_train - x, axis=1)
        k_idx = np.argsort(dist)[:self.k]
        k_labels = self.y_train[k_idx]
        vals, counts = np.unique(k_labels, return_counts=True)
        return vals[np.argmax(counts)]
    
    def predict(self, X): 
        return np.array([self.predict_one(x) for x in X])

class DecisionTree:
    def __init__(self, depth=5): 
        self.depth = depth
        self.tree = None
        self.name = "Decision Tree"
    
    def fit(self, X, y): 
        self.tree = self._build_tree(X, y, self.depth)
    
    def _build_tree(self, X, y, depth):
        if depth == 0 or len(np.unique(y)) == 1 or len(X) == 0:
            vals, counts = np.unique(y, return_counts=True)
            return 0 if len(counts) == 0 else vals[np.argmax(counts)]
        
        feat = np.random.randint(X.shape[1])
        threshold = np.median(X[:, feat])
        left_idx = X[:, feat] <= threshold
        right_idx = X[:, feat] > threshold
        
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            vals, counts = np.unique(y, return_counts=True)
            return vals[np.argmax(counts)]
        
        return (feat, threshold, 
                self._build_tree(X[left_idx], y[left_idx], depth-1),
                self._build_tree(X[right_idx], y[right_idx], depth-1))
    
    def predict_one(self, x, tree=None):
        if tree is None: 
            tree = self.tree
        if not isinstance(tree, tuple): 
            return tree
        feat, threshold, left, right = tree
        return self.predict_one(x, left) if x[feat] <= threshold else self.predict_one(x, right)
    
    def predict(self, X): 
        return np.array([self.predict_one(x) for x in X])

class SoftmaxRegression:
    def __init__(self, lr=0.01, epochs=300): 
        self.lr = lr
        self.epochs = epochs
        self.name = "Softmax Regression"
    
    def fit(self, X, y):
        X = X.astype(np.float64)
        self.classes = np.unique(y)
        self.W = np.zeros((X.shape[1], len(self.classes)))
        
        for _ in range(self.epochs):
            scores = X @ self.W
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            y_onehot = np.zeros_like(probs)
            y_onehot[np.arange(len(y)), y] = 1
            grad = X.T @ (probs - y_onehot) / X.shape[0]
            self.W -= self.lr * grad
    
    def predict(self, X):
        scores = X @ self.W
        return np.argmax(scores, axis=1)
    
    def predict_proba(self, X):
        scores = X @ self.W
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# Train models with progress indication
with st.spinner('🚀 Training advanced ML models...'):
    knn_model = KNN(k=5)
    knn_model.fit(X_train, y_train)
    
    dt_model = DecisionTree(depth=5)
    dt_model.fit(X_train, y_train)
    
    lr_model = SoftmaxRegression(lr=0.1, epochs=300)
    lr_model.fit(X_train, y_train)

# Generate predictions
y_knn = knn_model.predict(X_test)
y_dt = dt_model.predict(X_test)
y_lr = lr_model.predict(X_test)

def calc_accuracy(y_true, y_pred): 
    return np.mean(y_true == y_pred)

def create_confusion_matrix(y_true, y_pred, labels):
    """Create confusion matrix without sklearn"""
    n_classes = len(labels)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    return cm

# Calculate accuracies
acc_knn = calc_accuracy(y_test, y_knn)
acc_dt = calc_accuracy(y_test, y_dt)
acc_lr = calc_accuracy(y_test, y_lr)

# --------------------------
# Header Section
# --------------------------
st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🛰️ Satellite Signal Strength Classification</h1>
        <p class="subtitle">Advanced Machine Learning Dashboard for Signal Quality Prediction</p>
    </div>
""", unsafe_allow_html=True)

# --------------------------
# Main Navigation Tabs
# --------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "📊 Analytics", "🧠 Model Insights", "🔮 Predictions"])

# --------------------------
# Dashboard Tab
# --------------------------
with tab1:
    # Performance Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{acc_knn*100:.1f}%</div>
                <div class="metric-label">🎯 KNN Accuracy</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{acc_dt*100:.1f}%</div>
                <div class="metric-label">🌳 Decision Tree</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{acc_lr*100:.1f}%</div>
                <div class="metric-label">🧮 Softmax Regression</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Interactive Prediction Section
    st.markdown("""
        <div class="input-section">
            <h2 class="section-title">🔮 Real-time Signal Prediction</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        rain_att = st.slider(
            "🌧️ Rain Attenuation (dB)", 
            float(df['rain_attenuation_db'].min()), 
            float(df['rain_attenuation_db'].max()), 
            float(df['rain_attenuation_db'].mean()),
            help="Signal loss due to atmospheric rain"
        )
        
        cn_ratio = st.slider(
            "📡 C/N Ratio (dB)", 
            float(df['cn_ratio_db'].min()), 
            float(df['cn_ratio_db'].max()), 
            float(df['cn_ratio_db'].mean()),
            help="Carrier-to-noise ratio indicating signal quality"
        )
        
        distance = st.slider(
            "📏 Distance (m)", 
            float(df['distance_m'].min()), 
            float(df['distance_m'].max()), 
            float(df['distance_m'].mean()),
            help="Distance between satellite and receiver"
        )
    
    with col2:
        frequency = st.slider(
            "📻 Frequency (GHz)", 
            float(df['frequency_hz'].min() / 1e9), 
            float(df['frequency_hz'].max() / 1e9), 
            float(df['frequency_hz'].mean() / 1e9),
            step=0.1,
            help="Operating frequency in GHz"
        )
        
        orbit_type = st.selectbox(
            "🛰️ Orbit Type", 
            df['orbit_type'].unique(),
            help="LEO: Low Earth Orbit, MEO: Medium Earth Orbit, GEO: Geostationary"
        )
    
    # Make predictions
    input_df = pd.DataFrame({
        'rain_attenuation_db': [rain_att],
        'cn_ratio_db': [cn_ratio],
        'distance_m': [distance],
        'frequency_hz': [frequency * 1e9],  # Convert back to Hz
        'orbit_type': [orbit_type]
    })
    
    input_X = pd.get_dummies(input_df, drop_first=True).reindex(
        columns=feature_columns, fill_value=0
    ).values.astype(np.float64)
    
    # Display predictions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        knn_pred = uniques[knn_model.predict(input_X)[0]]
        st.markdown(f"""
            <div class="prediction-box">
                <strong>🎯 KNN:</strong> {knn_pred}
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        dt_pred = uniques[dt_model.predict(input_X)[0]]
        st.markdown(f"""
            <div class="prediction-box">
                <strong>🌳 Decision Tree:</strong> {dt_pred}
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        lr_pred = uniques[lr_model.predict(input_X)[0]]
        st.markdown(f"""
            <div class="prediction-box">
                <strong>🧮 Softmax LR:</strong> {lr_pred}
            </div>
        """, unsafe_allow_html=True)

# --------------------------
# Analytics Tab
# --------------------------
with tab2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Interactive Plotly scatter plot
    fig_scatter = px.scatter(
        df, 
        x='rain_attenuation_db', 
        y='cn_ratio_db',
        color='signal_quality',
        size='distance_m',
        hover_data=['frequency_hz', 'orbit_type'],
        title="Signal Quality Analysis: C/N Ratio vs Rain Attenuation",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig_scatter.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter"
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Distribution plot
        fig_dist = px.histogram(
            df, 
            x='signal_quality', 
            color='orbit_type',
            title="Signal Quality Distribution by Orbit Type",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        fig_dist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Inter"
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Box plot
        fig_box = px.box(
            df, 
            x='signal_quality', 
            y='cn_ratio_db',
            color='signal_quality',
            title="C/N Ratio Distribution by Signal Quality",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig_box.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Inter",
            showlegend=False
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --------------------------
# Model Insights Tab
# --------------------------
with tab3:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🔍 Confusion Matrices Comparison")
    
    # Create custom confusion matrices
    cm_knn = create_confusion_matrix(y_test, y_knn, uniques)
    cm_dt = create_confusion_matrix(y_test, y_dt, uniques)
    cm_lr = create_confusion_matrix(y_test, y_lr, uniques)
    
    # Create subplots for confusion matrices
    fig_cm = make_subplots(
        rows=1, cols=3,
        subplot_titles=('KNN', 'Decision Tree', 'Softmax Regression'),
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}]]
    )
    
    # Add heatmaps
    for i, (cm, title) in enumerate(zip([cm_knn, cm_dt, cm_lr], ['KNN', 'Decision Tree', 'Softmax LR'])):
        fig_cm.add_trace(
            go.Heatmap(
                z=cm,
                x=uniques,
                y=uniques,
                colorscale='turbo',
                showscale=i==2
            ),
            row=1, col=i+1
        )
    
    fig_cm.update_layout(
        title="Model Performance Comparison",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter"
    )
    
    st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model-specific insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📊 Prediction Confidence")
        
        # Softmax probabilities for visualization
        if len(X_test) > 0:
            probs = lr_model.predict_proba(X_test[:50])
            
            fig_probs = go.Figure()
            for i, label in enumerate(uniques):
                fig_probs.add_trace(go.Scatter(
                    y=probs[:, i],
                    mode='lines+markers',
                    name=label,
                    line=dict(width=3)
                ))
            
            fig_probs.update_layout(
                title="Softmax Prediction Probabilities (First 50 Samples)",
                xaxis_title="Sample Index",
                yaxis_title="Probability",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_family="Inter"
            )
            
            st.plotly_chart(fig_probs, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🎯 Model Accuracy Comparison")
        
        # Accuracy comparison
        model_names = ['KNN', 'Decision Tree', 'Softmax LR']
        accuracies = [acc_knn * 100, acc_dt * 100, acc_lr * 100]
        
        fig_acc = go.Figure(data=[
            go.Bar(
                x=model_names,
                y=accuracies,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                text=[f'{acc:.1f}%' for acc in accuracies],
                textposition='auto'
            )
        ])
        
        fig_acc.update_layout(
            title="Model Performance Comparison",
            yaxis_title="Accuracy (%)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Inter"
        )
        
        st.plotly_chart(fig_acc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --------------------------
# Predictions Tab
# --------------------------
with tab4:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🔮 Batch Prediction Analysis")
    
    # Sample predictions analysis
    sample_indices = np.random.choice(len(X_test), min(20, len(X_test)), replace=False)
    sample_data = []
    
    for idx in sample_indices:
        sample_data.append({
            'Sample': idx,
            'Actual': uniques[y_test[idx]],
            'KNN': uniques[y_knn[idx]],
            'Decision Tree': uniques[y_dt[idx]],
            'Softmax LR': uniques[y_lr[idx]]
        })
    
    pred_df = pd.DataFrame(sample_data)
    
    # Create comparison visualization
    fig_pred = go.Figure()
    
    # Add traces for each model
    models = ['KNN', 'Decision Tree', 'Softmax LR']
    colors_models = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, model in enumerate(models):
        # Calculate match with actual
        matches = (pred_df[model] == pred_df['Actual']).astype(int)
        fig_pred.add_trace(go.Bar(
            name=model,
            x=pred_df['Sample'],
            y=matches,
            marker_color=colors_models[i],
            opacity=0.7
        ))
    
    fig_pred.update_layout(
        title="Prediction Accuracy by Sample (1 = Correct, 0 = Incorrect)",
        xaxis_title="Sample Index",
        yaxis_title="Prediction Accuracy",
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter"
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Display detailed results table
    st.subheader("📋 Detailed Prediction Results")
    st.dataframe(
        pred_df,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------
# Footer
# --------------------------
st.markdown("""
    <div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.7);">
        <p>🛰️ Satellite Signal Analytics Dashboard | Built with Streamlit & Advanced ML</p>
        <p>
    Dhyey Savaliya<br>
  <a href="https://www.linkedin.com/in/dhyey-savaliya-632bb4246/" target="_blank">LinkedIn</a> 
  <a href="https://github.com/cursed-0men" target="_blank">GitHub</a>
    </div>
""", unsafe_allow_html=True)