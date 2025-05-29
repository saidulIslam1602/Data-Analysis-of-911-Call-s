"""
Advanced Visualization Module for 911 Emergency Calls Analysis

This module provides sophisticated plotting functions for:
- Time series analysis and forecasting visualization
- Geographic and spatial analysis plots
- Statistical and correlation analysis
- Machine learning model evaluation plots
- Interactive dashboard components
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import folium
from folium import plugins
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class EmergencyCallsVisualizer:
    """Advanced visualization class for emergency calls data"""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.color_palette = px.colors.qualitative.Set1
        
        # Set matplotlib style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def plot_time_series_analysis(self, df: pd.DataFrame, save_path: Optional[str] = None) -> go.Figure:
        """
        Create comprehensive time series analysis plots
        
        Args:
            df: DataFrame with timestamp column
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        # Prepare time series data
        daily_calls = df.groupby(df['timeStamp'].dt.date).size().reset_index()
        daily_calls.columns = ['Date', 'Call_Count']
        daily_calls['Date'] = pd.to_datetime(daily_calls['Date'])
        
        # Calculate rolling averages
        daily_calls['MA_7'] = daily_calls['Call_Count'].rolling(window=7).mean()
        daily_calls['MA_30'] = daily_calls['Call_Count'].rolling(window=30).mean()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Daily Call Volume', 'Call Volume Distribution',
                'Seasonal Decomposition', 'Moving Averages',
                'Hourly Patterns', 'Weekly Patterns'
            ),
            specs=[[{"colspan": 2}, None],
                   [{"colspan": 2}, None],
                   [{}, {}]]
        )
        
        # Daily call volume
        fig.add_trace(
            go.Scatter(
                x=daily_calls['Date'], y=daily_calls['Call_Count'],
                mode='lines', name='Daily Calls',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Moving averages
        fig.add_trace(
            go.Scatter(
                x=daily_calls['Date'], y=daily_calls['MA_7'],
                mode='lines', name='7-day MA',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_calls['Date'], y=daily_calls['MA_30'],
                mode='lines', name='30-day MA',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # Hourly patterns
        if 'hour' in df.columns:
            hourly_avg = df.groupby('hour').size().reset_index()
            hourly_avg.columns = ['Hour', 'Average_Calls']
            
            fig.add_trace(
                go.Bar(
                    x=hourly_avg['Hour'], y=hourly_avg['Average_Calls'],
                    name='Hourly Average', marker_color='lightblue'
                ),
                row=3, col=1
            )
        
        # Weekly patterns
        if 'dayofweek' in df.columns:
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            weekly_avg = df.groupby('dayofweek').size().reset_index()
            weekly_avg.columns = ['Day', 'Average_Calls']
            weekly_avg['Day_Name'] = [day_names[i] for i in weekly_avg['Day']]
            
            fig.add_trace(
                go.Bar(
                    x=weekly_avg['Day_Name'], y=weekly_avg['Average_Calls'],
                    name='Weekly Average', marker_color='lightgreen'
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Comprehensive Time Series Analysis",
            showlegend=True,
            title_x=0.5
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_geographic_analysis(self, df: pd.DataFrame, save_path: Optional[str] = None) -> folium.Map:
        """
        Create advanced geographic analysis map
        
        Args:
            df: DataFrame with lat/lng columns
            save_path: Optional path to save the map
            
        Returns:
            Folium map object
        """
        # Filter valid coordinates
        geo_df = df.dropna(subset=['lat', 'lng'])
        geo_df = geo_df[
            (geo_df['lat'].between(39, 42)) &
            (geo_df['lng'].between(-76, -74))
        ]
        
        if geo_df.empty:
            raise ValueError("No valid geographic data available")
        
        # Create base map
        center_lat = geo_df['lat'].mean()
        center_lng = geo_df['lng'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=10,
            tiles='CartoDB positron'
        )
        
        # Add heatmap layer
        if len(geo_df) > 1000:
            # Sample for performance
            heat_data = geo_df.sample(n=1000, random_state=42)[['lat', 'lng']].values.tolist()
        else:
            heat_data = geo_df[['lat', 'lng']].values.tolist()
        
        plugins.HeatMap(heat_data, radius=15, blur=10).add_to(m)
        
        # Add cluster markers by emergency type
        if 'emergency_category' in geo_df.columns:
            emergency_types = geo_df['emergency_category'].unique()
            colors = ['red', 'blue', 'green', 'purple', 'orange']
            
            for i, emergency_type in enumerate(emergency_types[:5]):
                type_data = geo_df[geo_df['emergency_category'] == emergency_type]
                
                # Sample for performance
                if len(type_data) > 200:
                    type_data = type_data.sample(n=200, random_state=42)
                
                marker_cluster = plugins.MarkerCluster(name=emergency_type).add_to(m)
                
                for idx, row in type_data.iterrows():
                    folium.CircleMarker(
                        location=[row['lat'], row['lng']],
                        radius=5,
                        popup=f"{emergency_type}<br>{row.get('addr', 'No address')}",
                        color=colors[i % len(colors)],
                        fill=True,
                        opacity=0.7
                    ).add_to(marker_cluster)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        if save_path:
            m.save(save_path)
        
        return m
    
    def plot_emergency_type_analysis(self, df: pd.DataFrame, save_path: Optional[str] = None) -> go.Figure:
        """
        Create comprehensive emergency type analysis
        
        Args:
            df: DataFrame with emergency category data
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        if 'emergency_category' not in df.columns:
            raise ValueError("Emergency category column not found")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Emergency Type Distribution',
                'Emergency Types by Hour',
                'Emergency Types by Day of Week',
                'Emergency Types by Month'
            ),
            specs=[[{"type": "domain"}, {}],
                   [{}, {}]]
        )
        
        # Emergency type distribution (pie chart)
        emergency_counts = df['emergency_category'].value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=emergency_counts.index,
                values=emergency_counts.values,
                name="Emergency Types"
            ),
            row=1, col=1
        )
        
        # Emergency types by hour
        if 'hour' in df.columns:
            hourly_emergency = df.groupby(['hour', 'emergency_category']).size().unstack(fill_value=0)
            
            for emergency_type in hourly_emergency.columns:
                fig.add_trace(
                    go.Scatter(
                        x=hourly_emergency.index,
                        y=hourly_emergency[emergency_type],
                        mode='lines+markers',
                        name=emergency_type,
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # Emergency types by day of week
        if 'dayofweek' in df.columns:
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            daily_emergency = df.groupby(['dayofweek', 'emergency_category']).size().unstack(fill_value=0)
            
            for emergency_type in daily_emergency.columns[:3]:  # Top 3 types
                fig.add_trace(
                    go.Bar(
                        x=[day_names[i] for i in daily_emergency.index],
                        y=daily_emergency[emergency_type],
                        name=emergency_type,
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # Emergency types by month
        if 'month' in df.columns:
            monthly_emergency = df.groupby(['month', 'emergency_category']).size().unstack(fill_value=0)
            
            for emergency_type in monthly_emergency.columns[:3]:  # Top 3 types
                fig.add_trace(
                    go.Scatter(
                        x=monthly_emergency.index,
                        y=monthly_emergency[emergency_type],
                        mode='lines+markers',
                        name=emergency_type,
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Emergency Type Analysis Dashboard",
            title_x=0.5
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, save_path: Optional[str] = None) -> go.Figure:
        """
        Create correlation heatmap for numerical features
        
        Args:
            df: DataFrame with numerical columns
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if not col.endswith('_encoded')]
        
        if len(numerical_cols) < 2:
            raise ValueError("Not enough numerical columns for correlation analysis")
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Feature Correlation Heatmap",
            title_x=0.5,
            width=800,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_model_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_prob: Optional[np.ndarray] = None, 
                            model_name: str = "Model",
                            save_path: Optional[str] = None) -> go.Figure:
        """
        Create comprehensive model evaluation plots
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for classification)
            model_name: Name of the model
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        # Determine if classification or regression
        is_classification = len(np.unique(y_true)) < 20 and y_prob is not None
        
        if is_classification:
            return self._plot_classification_evaluation(y_true, y_pred, y_prob, model_name, save_path)
        else:
            return self._plot_regression_evaluation(y_true, y_pred, model_name, save_path)
    
    def _plot_classification_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_prob: np.ndarray, model_name: str,
                                      save_path: Optional[str] = None) -> go.Figure:
        """Create classification evaluation plots"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Confusion Matrix',
                'ROC Curve',
                'Precision-Recall Curve',
                'Feature Importance'
            )
        )
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        fig.add_trace(
            go.Heatmap(
                z=cm,
                text=cm,
                texttemplate="%{text}",
                colorscale='Blues'
            ),
            row=1, col=1
        )
        
        # ROC Curve (for binary classification)
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob)
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC Curve (AUC = {roc_auc:.2f})',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=2
            )
            
            # Add diagonal line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(color='red', dash='dash'),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob)
            
            fig.add_trace(
                go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name='Precision-Recall',
                    line=dict(color='green', width=2)
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"{model_name} - Classification Evaluation",
            title_x=0.5
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def _plot_regression_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   model_name: str, save_path: Optional[str] = None) -> go.Figure:
        """Create regression evaluation plots"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Actual vs Predicted',
                'Residual Plot',
                'Residual Distribution',
                'Error Distribution'
            )
        )
        
        # Actual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y_true, y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(color='blue', opacity=0.6)
            ),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Residual plot
        residuals = y_true - y_pred
        fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='green', opacity=0.6)
            ),
            row=1, col=2
        )
        
        # Zero line
        fig.add_trace(
            go.Scatter(
                x=[y_pred.min(), y_pred.max()], y=[0, 0],
                mode='lines',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Residual distribution
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=30,
                name='Residual Distribution',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        # Error distribution
        errors = np.abs(residuals)
        fig.add_trace(
            go.Histogram(
                x=errors,
                nbinsx=30,
                name='Error Distribution',
                marker_color='lightcoral'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"{model_name} - Regression Evaluation",
            title_x=0.5
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], 
                              top_n: int = 15, save_path: Optional[str] = None) -> go.Figure:
        """
        Plot feature importance
        
        Args:
            feature_importance: Dictionary of feature names and importance scores
            top_n: Number of top features to display
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        if not feature_importance:
            raise ValueError("Feature importance dictionary is empty")
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        features, importances = zip(*top_features)
        
        # Create bar plot
        fig = go.Figure(data=[
            go.Bar(
                x=list(importances),
                y=list(features),
                orientation='h',
                marker_color='lightblue',
                text=[f'{imp:.3f}' for imp in importances],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f"Top {top_n} Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=600,
            title_x=0.5
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_statistical_summary(self, df: pd.DataFrame, save_path: Optional[str] = None) -> go.Figure:
        """
        Create comprehensive statistical summary dashboard
        
        Args:
            df: DataFrame to analyze
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Data Quality Overview',
                'Missing Data Pattern',
                'Numerical Features Distribution',
                'Categorical Features Distribution',
                'Temporal Patterns',
                'Geographic Distribution'
            ),
            specs=[[{"type": "domain"}, {}],
                   [{}, {}],
                   [{}, {}]]
        )
        
        # Data quality overview
        total_records = len(df)
        missing_records = df.isnull().any(axis=1).sum()
        complete_records = total_records - missing_records
        
        fig.add_trace(
            go.Pie(
                labels=['Complete Records', 'Records with Missing Data'],
                values=[complete_records, missing_records],
                name="Data Quality"
            ),
            row=1, col=1
        )
        
        # Missing data pattern
        missing_data = df.isnull().sum().sort_values(ascending=True)
        missing_data = missing_data[missing_data > 0]
        
        if not missing_data.empty:
            fig.add_trace(
                go.Bar(
                    x=missing_data.values,
                    y=missing_data.index,
                    orientation='h',
                    name='Missing Data Count',
                    marker_color='red'
                ),
                row=1, col=2
            )
        
        # Numerical features distribution
        numerical_cols = df.select_dtypes(include=[np.number]).columns[:5]  # Top 5
        for col in numerical_cols:
            if df[col].notna().sum() > 0:
                fig.add_trace(
                    go.Histogram(
                        x=df[col].dropna(),
                        name=col,
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Comprehensive Statistical Summary",
            title_x=0.5,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


def create_publication_plots(df: pd.DataFrame, output_dir: str = "plots/") -> None:
    """
    Create publication-ready plots for research papers
    
    Args:
        df: DataFrame with emergency calls data
        output_dir: Directory to save plots
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    visualizer = EmergencyCallsVisualizer()
    
    # Create various plots
    try:
        # Time series analysis
        time_fig = visualizer.plot_time_series_analysis(df)
        time_fig.write_html(output_path / "time_series_analysis.html")
        
        # Emergency type analysis
        if 'emergency_category' in df.columns:
            emergency_fig = visualizer.plot_emergency_type_analysis(df)
            emergency_fig.write_html(output_path / "emergency_type_analysis.html")
        
        # Correlation heatmap
        corr_fig = visualizer.plot_correlation_heatmap(df)
        corr_fig.write_html(output_path / "correlation_heatmap.html")
        
        # Statistical summary
        summary_fig = visualizer.create_statistical_summary(df)
        summary_fig.write_html(output_path / "statistical_summary.html")
        
        print(f"Publication plots saved to {output_path}")
        
    except Exception as e:
        print(f"Error creating plots: {e}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualization plots for 911 calls data')
    parser.add_argument('--data', required=True, help='Path to processed data file')
    parser.add_argument('--output', default='plots/', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load data
    if args.data.endswith('.parquet'):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)
    
    # Create plots
    create_publication_plots(df, args.output) 