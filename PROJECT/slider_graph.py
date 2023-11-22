# Import necessary libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
import dash
from dash.dependencies import Input, Output

# Load the datasets
df_cases = pd.read_csv('application\Cleaned Covid Data.csv', parse_dates=['date'])
df_equity = pd.read_csv('application\cleaned indices data.csv', parse_dates=['Date'])

# filter out all dates before 2018
df_cases = df_cases[df_cases['date'] >= pd.Timestamp('2018-01-01')]
df_equity = df_equity[df_equity['Date'] >= pd.Timestamp('2018-01-01')]

# Convert dates to numerical format for the slider
df_cases['date_num'] = (df_cases['date'] - pd.Timestamp('2020-01-01')).dt.days
df_equity['date_num'] = (df_equity['Date'] - pd.Timestamp('2020-01-01')).dt.days

# Define a function to get the start of the quarter for a given date
def get_quarter_start(date):
    return pd.Timestamp(year=date.year, month=((date.month-1)//3)*3+1, day=1)

# Get unique quarterly dates
quarterly_dates = df_cases['date'].apply(get_quarter_start).unique()

# Replace 0 values in the equity data with the previous day's value
df_equity = df_equity.replace(0, np.nan)
df_equity = df_equity.fillna(method='ffill')

# Initialize the Dash app
dash_app = dash.Dash(__name__)

# Define the app layout
dash_app.layout = html.Div([
    dcc.Slider(
        id='time-slider',
        min=df_cases['date_num'].min(),
        max=df_cases['date_num'].max(),
        value=df_cases['date_num'].min(),
        marks={str(date_num): pd.to_datetime(str(date)).strftime('%Y-%m-%d') for date_num, date in zip(df_cases[df_cases['date'].isin(quarterly_dates)]['date_num'], quarterly_dates)},
        step=1,
        updatemode='drag'
    ),
    dcc.Checklist(
        id='country-checklist',
        options=[{'label': 'Overall', 'value': 'Overall'}] + [{'label': country, 'value': country} for country in df_cases['location'].unique()],
        value=['Japan', 'China', 'Germany', 'United States'], #df_cases['location'].unique().tolist(),
        inline=True
    ),
    html.Div(style={'display': 'flex', 'height': '1000px'}, children=[
        html.Div(id='choropleth-div', style={'width': '50%', 'display': 'inline-block'}),
        html.Div(id='candlestick-div', style={'width': '50%', 'display': 'inline-block'})
    ])
])

# Define callback to update checklist
@dash_app.callback(
    Output('country-checklist', 'value'),
    [Input('country-checklist', 'value')]
)
def update_checklist(selected_values):
    all_countries = df_cases['location'].unique().tolist()
    if 'Overall' in selected_values:
        return ['Overall']
    elif len(selected_values) > 4:
        return selected_values[:4]  # Limit to the first 4 selected countries
    else:
        return selected_values

# Define callback to update graphs
@dash_app.callback(
    [Output('choropleth-div', 'children'),
     Output('candlestick-div', 'children')],
    [Input('time-slider', 'value'),
     Input('country-checklist', 'value')]
)
def update_figure(selected_date_num, selected_countries):
    if 'Overall' in selected_countries:
        filtered_df_cases = df_cases[df_cases['date_num'] == selected_date_num]
    else:
        filtered_df_cases = df_cases[(df_cases['date_num'] == selected_date_num) & (df_cases['location'].isin(selected_countries))]
    filtered_df_equity = df_equity[df_equity['date_num'] <= selected_date_num]

    fig_cases = px.choropleth(filtered_df_cases, locations='iso_code', color='total_cases',
                              title='Infected Cases by Country',
                              color_continuous_scale='Reds')
    fig_cases.update_layout(
        autosize=True,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_font=dict(size=14),  # Smaller font size for the title
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)  # Horizontal legend at the bottom
    )

    indices = {
        'NKY_index': 'Japan',
        'XIN9I_index': 'China',
        'DAX_index': 'Germany',
        'SPX_index': 'United States',
        'MAIN_MARKET_50_index': 'Kuwait',
        'NSE-ALL_SHARE_index': 'Nigeria',
        'NIFTY_50_index': 'India',
        'FTSE_AOX_GENERAL_index': 'United Arab Emirates',
        'QE_GENERAL_index': 'Qatar',
        'FTSE_100_index': 'United Kingdom',
        'BOVESPA_index': 'Brazil'
    }

    choropleth_graph = dcc.Graph(id='infected-cases-graph', figure=fig_cases, config={'displayModeBar': False})
    candlestick_graphs = []

    if 'Overall' in selected_countries:
        # Calculate the total value of all indices for each date
        total_value = filtered_df_equity[[f'{index}_Close' for index in indices.keys()]].groupby(filtered_df_equity['Date']).sum().sum(axis=1)
        fig_total_value = go.Figure(data=[go.Scatter(x=total_value.index, y=total_value.values, mode='lines', name='Total Value')])
        fig_total_value.update_layout(
            autosize=True,
            margin=dict(l=0, r=0, t=30, b=30),  # Added bottom margin
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title='Total Value of Indices of All Countries',
            title_font=dict(size=14),  # Smaller font size for the title
            yaxis=dict(range=[150000, 350000])  # Fixed y-axis range
        )
        fig_total_value.update_xaxes(range=['2018-01-01', '2022-12-31'])
        candlestick_graphs.append(dcc.Graph(id='total-value-graph', figure=fig_total_value, style={'height': '50%', 'display': 'block'}, config={'displayModeBar': False}))
    else:
        # Calculate the number of selected countries
        num_countries = len(selected_countries)

        # Set the height of the candlestick charts
        if num_countries > 0:
            height = 50 / num_countries
        else:
            height = 50  # or any other default value

        for index, country in indices.items():
            if country in selected_countries:
                fig_equity = go.Figure(data=[go.Candlestick(x=filtered_df_equity['Date'],
                            open=filtered_df_equity[f'{index}_Open'],
                            high=filtered_df_equity[f'{index}_High'],
                            low=filtered_df_equity[f'{index}_Low'],
                            close=filtered_df_equity[f'{index}_Close'],
                            name=country)])
                fig_equity.update_layout(
                    autosize=True,
                    margin=dict(l=0, r=0, t=60, b=30),  # Added bottom margin
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    title=f'Equity Index for {country}',
                    title_font=dict(size=10)  # Smaller font size for the title
                )
                fig_equity.update_xaxes(range=['2018-01-01', '2022-12-31'])
                candlestick_graphs.append(dcc.Graph(id=f'{country}-equity-index-graph', figure=fig_equity, style={'height': f'{height}%', 'display': 'block'}, config={'displayModeBar': False}))

    return choropleth_graph, candlestick_graphs

# Run the app
if __name__ == '_main_':
    dash_app.run_server(debug=True)