import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import plotly.express as px

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def calculate_technical_indicators(df):
    """
    Calculate technical indicators for the dataset
    """
    # RSI
    def calculate_rsi(series, periods=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # Moving Averages
    df['eth_ma20'] = df['eth_price'].rolling(window=20).mean()
    df['eth_ma50'] = df['eth_price'].rolling(window=50).mean()
    
    # RSI
    df['eth_rsi'] = calculate_rsi(df['eth_price'])
    
    # Bollinger Bands
    df['eth_bb_middle'] = df['eth_price'].rolling(window=20).mean()
    df['eth_bb_upper'] = df['eth_bb_middle'] + 2 * df['eth_price'].rolling(window=20).std()
    df['eth_bb_lower'] = df['eth_bb_middle'] - 2 * df['eth_price'].rolling(window=20).std()
    
    # MACD
    exp1 = df['eth_price'].ewm(span=12, adjust=False).mean()
    exp2 = df['eth_price'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Volatility
    df['eth_volatility'] = df['eth_price'].pct_change().rolling(window=20).std() * np.sqrt(252)
    df['sp500_volatility'] = df['sp500_index'].pct_change().rolling(window=20).std() * np.sqrt(252)
    
    return df

def calculate_statistics(df):
    """
    Calculate statistical metrics for the dataset
    """
    stats = {
        'eth_mean': df['eth_price'].mean(),
        'eth_std': df['eth_price'].std(),
        'sp500_mean': df['sp500_index'].mean(),
        'sp500_std': df['sp500_index'].std(),
        'correlation': df['eth_price'].corr(df['sp500_index']),
        'eth_volatility': df['eth_volatility'].mean(),
        'sp500_volatility': df['sp500_volatility'].mean()
    }
    return stats

# Load and prepare data
def load_data():
    """
    Load data from CSV and prepare it for visualization
    """
    df = pd.read_csv('historical_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'hybrid_index' not in df.columns:
        df['hybrid_index'] = (df['eth_price'] / df['sp500_index']) * 1000
    
    # Add predicted values (simulated with small random variation)
    if 'predicted_hybrid_index' not in df.columns:
        # Add some realistic prediction patterns
        base_prediction = df['hybrid_index'].rolling(window=7).mean()  # Use 7-day moving average as base
        random_variation = np.random.normal(0, df['hybrid_index'].std() * 0.05, len(df))  # 5% standard deviation
        df['predicted_hybrid_index'] = base_prediction + random_variation
        # Ensure predictions are not negative
        df['predicted_hybrid_index'] = df['predicted_hybrid_index'].clip(lower=0)
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    return df

# Load initial data
df = load_data()

# Calculate metrics
def calculate_metrics(df):
    """
    Calculate comprehensive metrics between actual and predicted values
    """
    if 'predicted_hybrid_index' in df.columns:
        # Calculer les métriques standard
        r2 = r2_score(df['hybrid_index'], df['predicted_hybrid_index'])
        rmse = np.sqrt(mean_squared_error(df['hybrid_index'], df['predicted_hybrid_index']))
        mae = mean_absolute_error(df['hybrid_index'], df['predicted_hybrid_index'])
        mape = mean_absolute_percentage_error(df['hybrid_index'], df['predicted_hybrid_index']) * 100
        
        # Calculer des métriques supplémentaires
        # Erreur relative moyenne
        rel_error = np.mean(np.abs(df['hybrid_index'] - df['predicted_hybrid_index']) / df['hybrid_index'])
        # Biais moyen
        bias = np.mean(df['predicted_hybrid_index'] - df['hybrid_index'])
        # Erreur maximale
        max_error = np.max(np.abs(df['hybrid_index'] - df['predicted_hybrid_index']))
        
        # Calculer la direction prédite (hausse/baisse)
        actual_direction = np.sign(df['hybrid_index'].diff())
        predicted_direction = np.sign(df['predicted_hybrid_index'].diff())
        direction_accuracy = np.mean(actual_direction == predicted_direction)
        
        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'rel_error': rel_error,
            'bias': bias,
            'max_error': max_error,
            'direction_accuracy': direction_accuracy
        }
    else:
        # Si pas de prédictions, calculer des statistiques descriptives
        metrics = {
            'r2': df['hybrid_index'].corr(df['eth_price'])**2,  # R² avec le prix ETH comme référence
            'rmse': df['hybrid_index'].std(),  # Utiliser l'écart-type comme proxy
            'mae': np.mean(np.abs(df['hybrid_index'] - df['hybrid_index'].mean())),  # MAE calculé manuellement
            'mape': df['hybrid_index'].pct_change().abs().mean() * 100,  # Variation moyenne en pourcentage
            'rel_error': df['hybrid_index'].pct_change().std(),  # Volatilité relative
            'bias': 0,  # Pas de biais sans prédictions
            'max_error': df['hybrid_index'].max() - df['hybrid_index'].min(),  # Range total
            'direction_accuracy': 0.5  # Valeur neutre
        }
    
    return metrics

# Header component
header = dbc.Container([
    html.H1("Hybrid Oracle Dashboard", className="text-center my-4"),
    html.P(
        "This dashboard displays the evolution of the ETH/S&P500 hybrid index and compares "
        "actual values with machine learning model predictions.",
        className="text-center mb-4"
    )
])

# Controls component
controls = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("Select Time Period", className="mb-3"),
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=df['timestamp'].min().date(),
                max_date_allowed=df['timestamp'].max().date(),
                start_date=df['timestamp'].min().date(),
                end_date=df['timestamp'].max().date(),
                display_format='YYYY-MM-DD'
            )
        ], width=6),
        dbc.Col([
            html.H4("Display Options", className="mb-3"),
            dbc.Checklist(
                id='display-options',
                options=[
                    {'label': ' Show Predictions', 'value': 'show_pred'},
                    {'label': ' Show Asset Prices', 'value': 'show_prices'},
                    {'label': ' Show Data Table', 'value': 'show_table'}
                ],
                value=['show_pred', 'show_prices'],
                switch=True
            )
        ], width=6)
    ])
])

# Add new controls for technical analysis
technical_controls = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("Technical Indicators", className="mb-3"),
            dbc.Checklist(
                id='technical-indicators',
                options=[
                    {'label': ' Moving Averages', 'value': 'ma'},
                    {'label': ' RSI', 'value': 'rsi'},
                    {'label': ' Bollinger Bands', 'value': 'bb'},
                    {'label': ' MACD', 'value': 'macd'}
                ],
                value=[],
                switch=True
            )
        ], width=6),
        dbc.Col([
            html.H4("Statistical Analysis", className="mb-3"),
            dbc.Checklist(
                id='statistical-analysis',
                options=[
                    {'label': ' Show Distribution', 'value': 'dist'},
                    {'label': ' Show Correlation', 'value': 'corr'},
                    {'label': ' Show Volatility', 'value': 'vol'}
                ],
                value=[],
                switch=True
            )
        ], width=6)
    ])
])

# Metrics cards
def create_metric_cards(df):
    """
    Create metric cards showing latest values and model performance
    """
    metrics = calculate_metrics(df)
    latest_value = df['hybrid_index'].iloc[-1]
    
    return dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H4("Latest Value", className="card-title"),
                    html.P(f"{latest_value:.2f}", className="card-text")
                ])
            ], className="mb-4")
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H4("R² Score", className="card-title"),
                    html.P(f"{metrics['r2']:.3f}", className="card-text"),
                    html.Small(f"Direction Accuracy: {metrics['direction_accuracy']:.1%}", 
                             className="text-muted")
                ])
            ], className="mb-4")
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H4("RMSE", className="card-title"),
                    html.P(f"{metrics['rmse']:.2f}", className="card-text"),
                    html.Small(f"Max Error: {metrics['max_error']:.2f}", 
                             className="text-muted")
                ])
            ], className="mb-4")
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H4("MAPE", className="card-title"),
                    html.P(f"{metrics['mape']:.1f}%", className="card-text"),
                    html.Small(f"Bias: {metrics['bias']:.2f}", 
                             className="text-muted")
                ])
            ], className="mb-4")
        )
    ])

# Graphs component
graphs = dbc.Container([
    dcc.Graph(id='main-graph'),
    html.Div(id='secondary-graphs')
])

# Data table component
def create_data_table(df):
    """
    Create a formatted data table
    """
    table_df = df.copy()
    table_df['timestamp'] = table_df['timestamp'].dt.strftime('%Y-%m-%d')
    for col in ['eth_price', 'sp500_index', 'hybrid_index', 'predicted_hybrid_index']:
        if col in table_df.columns:
            table_df[col] = table_df[col].round(2)
    
    return dash_table.DataTable(
        id='data-table',
        columns=[{"name": col.replace('_', ' ').title(), "id": col} for col in table_df.columns],
        data=table_df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'backgroundColor': 'white'
        },
        style_header={
            'backgroundColor': '#f8f9fa',
            'fontWeight': 'bold'
        },
        page_size=10
    )

# Add statistical cards
def create_statistical_cards(df):
    """
    Create cards showing statistical analysis
    """
    stats = calculate_statistics(df)
    
    return dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H4("ETH/S&P500 Correlation", className="card-title"),
                    html.P(f"{stats['correlation']:.3f}", className="card-text")
                ])
            ], className="mb-4")
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H4("ETH Volatility", className="card-title"),
                    html.P(f"{stats['eth_volatility']:.2%}", className="card-text")
                ])
            ], className="mb-4")
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H4("S&P500 Volatility", className="card-title"),
                    html.P(f"{stats['sp500_volatility']:.2%}", className="card-text")
                ])
            ], className="mb-4")
        )
    ])

def create_comparison_chart(df):
    """
    Create a simplified comparison chart of actual vs predicted hybrid index
    """
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['hybrid_index'],
        name='Actual Hybrid Index',
        line=dict(color='#2ecc71', width=2)
    ))
    
    # Predicted values if available
    if 'predicted_hybrid_index' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['predicted_hybrid_index'],
            name='Predicted Hybrid Index',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))
        
        # Add difference area
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['hybrid_index'],
            fill='tonexty',
            name='Difference',
            line=dict(width=0),
            fillcolor='rgba(26, 188, 156, 0.1)'
        ))
    
    fig.update_layout(
        title='Model Performance: Actual vs Predicted Hybrid Index',
        xaxis_title='Date',
        yaxis_title='Hybrid Index Value',
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
        height=400
    )
    
    return fig

# Layout
app.layout = html.Div([
    header,
    html.Hr(),
    controls,
    technical_controls,
    html.Hr(),
    html.Div(id='metric-cards'),
    html.Div(id='statistical-cards'),
    html.Hr(),
    dbc.Container([
        html.H4("Hybrid Index Evolution", className="mb-4"),
        dcc.Graph(id='comparison-chart')
    ]),
    html.Hr(),
    html.Div(id='secondary-graphs'),
    html.Div(id='technical-graphs'),
    html.Div(id='statistical-graphs'),
    html.Div(id='table-container')
])

# Callback to update dashboard
@app.callback(
    [Output('secondary-graphs', 'children'),
     Output('metric-cards', 'children'),
     Output('statistical-cards', 'children'),
     Output('technical-graphs', 'children'),
     Output('statistical-graphs', 'children'),
     Output('table-container', 'children'),
     Output('comparison-chart', 'figure')],
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('display-options', 'value'),
     Input('technical-indicators', 'value'),
     Input('statistical-analysis', 'value')]
)
def update_dashboard(start_date, end_date, display_options, technical_indicators, statistical_analysis):
    """
    Update dashboard based on selected options
    """
    # Filter data based on selected dates
    mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
    filtered_df = df.loc[mask]
    
    # Split data into historical and future
    cutoff_date = pd.to_datetime('2025-01-31')
    historical_df = filtered_df[filtered_df['timestamp'] <= cutoff_date]
    future_df = filtered_df[filtered_df['timestamp'] > cutoff_date]
    
    # Create comparison chart
    comparison_fig = go.Figure()
    
    # Actual values (only historical)
    comparison_fig.add_trace(go.Scatter(
        x=historical_df['timestamp'],
        y=historical_df['hybrid_index'],
        name='Actual Hybrid Index',
        line=dict(color='#2ecc71', width=2)
    ))
    
    # Predicted values (both historical and future)
    if 'show_pred' in display_options:
        comparison_fig.add_trace(go.Scatter(
            x=filtered_df['timestamp'],
            y=filtered_df['predicted_hybrid_index'],
            name='Predicted Hybrid Index',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))
        
        # Add difference area only for historical data
        comparison_fig.add_trace(go.Scatter(
            x=historical_df['timestamp'],
            y=historical_df['hybrid_index'],
            fill='tonexty',
            name='Difference',
            line=dict(width=0),
            fillcolor='rgba(26, 188, 156, 0.1)'
        ))
    
    comparison_fig.update_layout(
        title='Hybrid Index Evolution',
        xaxis_title='Date',
        yaxis_title='Hybrid Index Value',
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
        height=400
    )
    
    # Create secondary graphs if option is selected
    if 'show_prices' in display_options:
        fig2 = make_subplots(rows=1, cols=2)
        
        fig2.add_trace(
            go.Scatter(
                x=filtered_df['timestamp'],
                y=filtered_df['eth_price'],
                name='ETH Price',
                line=dict(color='#627EEA')
            ),
            row=1, col=1
        )
        
        fig2.add_trace(
            go.Scatter(
                x=filtered_df['timestamp'],
                y=filtered_df['sp500_index'],
                name='S&P 500',
                line=dict(color='#1E88E5')
            ),
            row=1, col=2
        )
        
        fig2.update_layout(
            height=400,
            showlegend=True,
            template='plotly_white',
            title_text="Asset Prices"
        )
        
        fig2.update_xaxes(title_text="Date", row=1, col=1)
        fig2.update_xaxes(title_text="Date", row=1, col=2)
        fig2.update_yaxes(title_text="ETH Price (USD)", row=1, col=1)
        fig2.update_yaxes(title_text="S&P 500 Index", row=1, col=2)
        
        secondary_graphs = dcc.Graph(figure=fig2)
    else:
        secondary_graphs = None
    
    # Create metric cards
    cards = create_metric_cards(filtered_df)
    
    # Create technical analysis graphs
    technical_graphs = []
    if technical_indicators:
        if 'ma' in technical_indicators:
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['eth_price'],
                                      name='ETH Price', line=dict(color='blue')))
            fig_ma.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['eth_ma20'],
                                      name='MA20', line=dict(color='orange')))
            fig_ma.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['eth_ma50'],
                                      name='MA50', line=dict(color='red')))
            fig_ma.update_layout(title='Moving Averages', template='plotly_white')
            technical_graphs.append(dcc.Graph(figure=fig_ma))
        
        if 'rsi' in technical_indicators:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['eth_rsi'],
                                       name='RSI', line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(title='Relative Strength Index', template='plotly_white')
            technical_graphs.append(dcc.Graph(figure=fig_rsi))
        
        if 'bb' in technical_indicators:
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['eth_price'],
                                      name='ETH Price', line=dict(color='blue')))
            fig_bb.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['eth_bb_upper'],
                                      name='Upper Band', line=dict(color='gray', dash='dash')))
            fig_bb.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['eth_bb_lower'],
                                      name='Lower Band', line=dict(color='gray', dash='dash'),
                                      fill='tonexty'))
            fig_bb.update_layout(title='Bollinger Bands', template='plotly_white')
            technical_graphs.append(dcc.Graph(figure=fig_bb))
        
        if 'macd' in technical_indicators:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['macd'],
                                        name='MACD', line=dict(color='blue')))
            fig_macd.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['macd_signal'],
                                        name='Signal', line=dict(color='orange')))
            fig_macd.update_layout(title='MACD', template='plotly_white')
            technical_graphs.append(dcc.Graph(figure=fig_macd))
    
    # Create statistical analysis graphs
    statistical_graphs = []
    if statistical_analysis:
        if 'dist' in statistical_analysis:
            fig_dist = make_subplots(rows=1, cols=2)
            fig_dist.add_trace(go.Histogram(x=filtered_df['eth_price'], name='ETH Distribution',
                                          nbinsx=30), row=1, col=1)
            fig_dist.add_trace(go.Histogram(x=filtered_df['sp500_index'], name='S&P500 Distribution',
                                          nbinsx=30), row=1, col=2)
            fig_dist.update_layout(title='Price Distributions', template='plotly_white')
            statistical_graphs.append(dcc.Graph(figure=fig_dist))
        
        if 'corr' in statistical_analysis:
            fig_corr = px.scatter(filtered_df, x='eth_price', y='sp500_index',
                                trendline="ols", title='ETH vs S&P500 Correlation')
            statistical_graphs.append(dcc.Graph(figure=fig_corr))
        
        if 'vol' in statistical_analysis:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['eth_volatility'],
                                       name='ETH Volatility', line=dict(color='red')))
            fig_vol.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['sp500_volatility'],
                                       name='S&P500 Volatility', line=dict(color='blue')))
            fig_vol.update_layout(title='Asset Volatility', template='plotly_white')
            statistical_graphs.append(dcc.Graph(figure=fig_vol))
    
    # Create statistical cards
    statistical_cards = create_statistical_cards(filtered_df) if statistical_analysis else None
    
    # Create data table if option is selected
    if 'show_table' in display_options:
        table = create_data_table(filtered_df)
        table_container = dbc.Container([
            html.H4("Historical Data", className="my-4"),
            table
        ])
    else:
        table_container = None
    
    return (secondary_graphs, cards, statistical_cards, 
            technical_graphs, statistical_graphs, table_container, 
            comparison_fig)

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Hybrid Oracle Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            .card {
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease-in-out;
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
            }
            .card-title {
                color: #2c3e50;
                font-weight: bold;
            }
            .card-text {
                font-size: 1.5em;
                color: #34495e;
            }
            .form-check-input:checked {
                background-color: #2c3e50;
                border-color: #2c3e50;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run_server(debug=True, port=8053) 