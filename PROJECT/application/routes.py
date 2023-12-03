from application import flask_app 
from slider_graph import dash_app
from flask import render_template,url_for
from markupsafe import Markup
import pandas as pd 
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
import numpy as np
import dash
from dash import dcc, html
import dash
from dash.dependencies import Input, Output
import bar_chart_race as bcr
import requests
from bs4 import BeautifulSoup


from application.dash import create_dash_application

import subprocess

# Define the FFmpeg download URL
ffmpeg_url = "https://www.ffmpeg.org/download.html"

# Define the command to download FFmpeg
download_command = f"wget {ffmpeg_url}"

# Run the download command using subprocess
subprocess.run(download_command, shell=True)
# Converting dash to an instance of flask
# server_dash = dash_app.server

# create_dash_application(flask_app)





def create_dash_application(flask_app):
    
   # Load the datasets
    df_cases = pd.read_csv('application\Cleaned Covid Data.csv', parse_dates=['date'])
    df_equity = pd.read_csv('application\Cleaned Indices Data.csv', parse_dates=['Date'])

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
    ## Creating dash app 
    # external_css=url_for('static',filename='main.css')
    dash_app=dash.Dash(server=flask_app,name='Dash_1',url_base_pathname='/dash1/',external_stylesheets=["static/main.css"])
    
    
    # dash_app.css.append_css({
    # "external_url": "main.css"
    # })    
    navbar = html.Nav(
        className="navbar navbar-expand-lg navbar-dark bg-primary",
        children=[
            html.Div(
                className="container-fluid",
                children=[
                    html.A(className="navbar-brand", href="#", children="COMP 4462"),
                    html.Button(
                        className="navbar-toggler",
                        type="button",
                        **{"data-bs-toggle": "collapse", "data-bs-target": "#navbarText", "aria-controls": "navbarText", "aria-expanded": "false", "aria-label": "Toggle navigation"},
                        children=[
                            html.Span(className="navbar-toggler-icon")
                        ]
                    ),
                    html.Div(
                        className="collapse navbar-collapse",
                        id="navbarText",
                        children=[
                            html.Ul(
                                className="navbar-nav ml-auto mb-2 mb-lg-0",
                                children=[
                                    html.Li(className="nav-item", children=[
                                        html.A(className="nav-link", href="/", children="Home")
                                    ]),
                                    html.Li(className="nav-item", children=[
                                        html.A(className="nav-link", href="/dash1/", children="Stock Indices")
                                    ]),
                                    html.Li(className="nav-item", children=[
                                        html.A(className="nav-link", href="/dash2/", children="Bivariate Map")
                                    ])
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )

# Initialize the Dash app
# dash_app = dash.Dash(__name__)

    # Define the app layout
    dash_app.layout = html.Div([
          
           
        html.H1("COVID-19 and Stock Indices"),
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
            inline=True,
            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
        ),
        dcc.RadioItems(
            id='yaxis-type',
            options=[{'label': i, 'value': i} for i in ['Zero', 'Minimum']],
            value='Zero',
            labelStyle={'display': 'inline-block'}
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
        elif len(selected_values) > 3:
            return selected_values[:3]  # Limit to the first 2 selected countries
        else:
            return selected_values

    # Define callback to update graphs
    @dash_app.callback(
        [Output('choropleth-div', 'children'),
        Output('candlestick-div', 'children')],
        [Input('time-slider', 'value'),
        Input('country-checklist', 'value'),
        Input('yaxis-type', 'value')]
    )
    def update_figure(selected_date_num, selected_countries, yaxis_type):
        if 'Overall' in selected_countries:
            filtered_df_cases = df_cases[df_cases['date_num'] == selected_date_num]
        else:
            filtered_df_cases = df_cases[(df_cases['date_num'] == selected_date_num) & (df_cases['location'].isin(selected_countries))]
        filtered_df_equity = df_equity[df_equity['date_num'] <= selected_date_num]

        fig_cases = go.Choropleth(
            locations=filtered_df_cases['iso_code'],
            z=filtered_df_cases['total_cases'],
            text=filtered_df_cases['location'],
            colorscale='Reds',
            marker_line_color='black',  # Set to white to make a clear distinction between countries
            marker_line_width=0.5,  # Thin lines
            name='Countries'
        )

        fig_cases_selected = go.Choropleth(
            locations=filtered_df_cases[filtered_df_cases['location'].isin(selected_countries)]['iso_code'],
            z=filtered_df_cases[filtered_df_cases['location'].isin(selected_countries)]['total_cases'],
            text=filtered_df_cases[filtered_df_cases['location'].isin(selected_countries)]['location'],
            colorscale='Reds',
            showscale=False,
            marker_line_color='black',  # Highlight selected countries in black
            marker_line_width=1,  # Thicker lines for selected countries
            name='Selected Countries'
        )

        layout = go.Layout(
            title_text='Infected Cases by Country',
            geo=dict(showframe=False, showcoastlines=False, projection_type='equirectangular'),
            annotations=[dict(x=0.55, y=0.1, xref='paper', yref='paper', text='', showarrow=False)]
        )

        fig = go.Figure(data=[fig_cases, fig_cases_selected], layout=layout)

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

        choropleth_graph = dcc.Graph(id='infected-cases-graph', figure=fig, config={'displayModeBar': False})
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

                    if yaxis_type == 'Zero':
                        fig_equity.update_yaxes(range=[0, filtered_df_equity[f'{index}_High'].max()])
                    else:
                        fig_equity.update_yaxes(range=[filtered_df_equity[f'{index}_Low'].min(), filtered_df_equity[f'{index}_High'].max()])

                    candlestick_graphs.append(dcc.Graph(id=f'{country}-equity-index-graph', figure=fig_equity, style={'height': f'{height}%', 'display': 'block'}, config={'displayModeBar': False}))

        return choropleth_graph, candlestick_graphs

    
    return dash_app
# # Run the app
# if __name__ == '_main_':
#     dash_app.run_server(debug=True)

def bivariate_map(flask_app):
  

    # Your helper functions for color interpolation
    # hex_to_rgb, rgb_to_hex, interpolate_color
    def hex_to_rgb(hex_color):
        """Convert a hex color to an RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def rgb_to_hex(rgb_color):
        """Convert an RGB tuple to a hex color."""
        return f'#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}'

    def interpolate_color(color1, color2, factor: float):
        """Interpolate between two hex colors."""
        rgb1 = hex_to_rgb(color1)
        rgb2 = hex_to_rgb(color2)
        blended_rgb = tuple(int(rgb1[i] + factor * (rgb2[i] - rgb1[i])) for i in range(3))
        return rgb_to_hex(blended_rgb)

    def create_color_legend(fig, x_start, y_start, cell_size, n_steps, label_x, label_y, label_offset):
        for i in range(n_steps):
            for j in range(n_steps):
                # Interpolate colors
                infected_color = interpolate_color('#dddddd', '#016eae', i / (n_steps - 1))
                gdp_color = interpolate_color('#dddddd', '#cc0024', j / (n_steps - 1))
                blended_color = interpolate_color(infected_color, gdp_color, 0.5)

                # Add a square cell with the interpolated color
                fig.add_shape(type='rect',
                            x0=x_start + i * cell_size, y0=y_start - j * cell_size,
                            x1=x_start + (i + 1) * cell_size, y1=y_start - (j + 1) * cell_size,
                            line=dict(color='black', width=1),
                            fillcolor=blended_color)

        # Add labels for the grid
        for i in range(n_steps):
            fig.add_annotation(x=x_start + i * cell_size + cell_size / 2, 
                            y=y_start + cell_size + 0.5*label_offset, 
                            text=str(round(i / (n_steps - 1), 1)),
                            showarrow=False, 
                            xanchor="center")
        for j in range(n_steps):
            fig.add_annotation(x=x_start - cell_size - 0.1*label_offset, 
                            y=y_start - j * cell_size - cell_size / 2, 
                            text=str(round(j / (n_steps - 1), 1)),
                            showarrow=False, 
                            yanchor="middle")

        # Add overall labels
        fig.add_annotation(x=x_start + (n_steps * cell_size / 2), 
                        y=y_start + cell_size + 2 * label_offset, 
                        text=label_x, 
                        showarrow=False, 
                        xanchor="center")
        fig.add_annotation(x=x_start - cell_size -  label_offset, 
                        y=y_start - (n_steps * cell_size / 2), 
                        text=label_y, 
                        showarrow=False, 
                        textangle=-90, 
                        yanchor="middle")


    # Load your dataset
    merged_df = pd.read_csv('application\merged_data.csv')
    #print(merged_df.isna().sum())
    merged_df = merged_df.dropna()
    # print(merged_df.isna().sum())
    
    # Initialize the Dash app
    # external_css=url_for('static',filename='main.css')
    app = dash.Dash(server=flask_app,name='Dash_2',url_base_pathname='/dash2/',external_stylesheets=["/static/main.css"])
        # Define the navbar HTML
    # app.css.append_css({
    # "external_url": "static/main.css"
    # })    
    navbar = html.Nav(
        className="navbar navbar-expand-lg navbar-dark bg-primary",
        children=[
            html.Div(
                className="container-fluid",
                children=[
                    html.A(className="navbar-brand", href="#", children="COMP 4462"),
                    html.Button(
                        className="navbar-toggler",
                        type="button",
                        **{"data-bs-toggle": "collapse", "data-bs-target": "#navbarText", "aria-controls": "navbarText", "aria-expanded": "false", "aria-label": "Toggle navigation"},
                        children=[
                            html.Span(className="navbar-toggler-icon")
                        ]
                    ),
                    html.Div(
                        className="collapse navbar-collapse",
                        id="navbarText",
                        children=[
                            html.Ul(
                                className="navbar-nav ml-auto mb-2 mb-lg-0",
                                children=[
                                    html.Li(className="nav-item", children=[
                                        html.A(className="nav-link", href="/", children="Home")
                                    ]),
                                    html.Li(className="nav-item", children=[
                                        html.A(className="nav-link", href="/dash1/", children="Stock Indices")
                                    ]),
                                    html.Li(className="nav-item", children=[
                                        html.A(className="nav-link", href="/dash2/", children="Bivariate Map")
                                    ])
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )

    # Define the app layout
    app.layout = html.Div([
    
        html.H1("COVID-19 Cases and GDP Analysis"),
        
        html.Label('Select Year:'),
        dcc.Slider(
            id='year-slider',
            min=2020,
            max=2022,
            value=2020,
            marks={str(year): str(year) for year in range(2020, 2023)},
            step=None
        ),

        html.Br(),

        html.Label('Select Infected Cases Threshold:'),
        dcc.Slider(
            id='cases-slider',
            min=0,
            max=merged_df[['2020_case', '2021_case', '2022_case']].max().max(),
            value=0,
            step=None
        ),

        dcc.Graph(id='bivariate-map')
    ])

    # Callback to update the bivariate map
    @app.callback(
        Output('bivariate-map', 'figure'),
        [Input('year-slider', 'value'), 
        Input('cases-slider', 'value')]
    )
    def update_bivariate_map(selected_year, selected_cases):
        year_str = str(selected_year)
        filtered_df = merged_df[merged_df[year_str + '_case'] >= selected_cases]

        fig = go.Figure()
    # Inside your callback after creating the main figure
        create_color_legend(fig, x_start=0.8, y_start=0.2, cell_size=0.04, n_steps=5, label_x="Infected Cases", label_y="GDP", label_offset=0.04)


        for _, row in filtered_df.iterrows():
            infected_cases_normalized = row[year_str + '_case'] / filtered_df[year_str + '_case'].max()
            gdp_normalized = row[year_str + '_gdp'] / filtered_df[year_str + '_gdp'].max()

            infected_color = interpolate_color('#dddddd', '#016eae', infected_cases_normalized)
            gdp_color = interpolate_color('#dddddd', '#cc0024', gdp_normalized)
            blended_color = interpolate_color(infected_color, gdp_color, 0.5)

            fig.add_trace(go.Choropleth(
                locations=[row['Country Code']],
                z=[row[year_str + '_case']],
                text=row['Country Name'],
                colorscale=[[0, blended_color], [1, blended_color]],
                showscale=False,
                marker_line_color='darkgray',
                marker_line_width=0.5
            ))

        fig.update_geos(projection_type="natural earth")
        fig.update_layout(
            title_text=f'COVID-19 Cases and GDP in {selected_year} (Cases >= {selected_cases})',
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth'
            )
        )

        return fig
    return app

## Calling the dash app 

dash_app = create_dash_application(flask_app)
dash_bivariate=bivariate_map(flask_app)

@flask_app.route("/")
def index(): 
    
    ## Overview Graph 1

    # Data
    df=pd.read_csv('application\Cleaned Covid Data.csv')
    
    df['date']=pd.to_datetime(df['date'])
    countries = df['location'].unique()
    # Filter rows where the date is the end of the month
    df_end_of_month = df[df['date'] + pd.offsets.MonthEnd(0) == df['date']]
    
    # Extract month and year from the date column
    df_end_of_month['month'] = df_end_of_month['date'].dt.strftime('%b %Y')

    # Get data up to February 2022
    df_end_of_month = df_end_of_month[(df_end_of_month['date'] <= pd.to_datetime('2022-02-28')) & (df_end_of_month['date'] >= pd.to_datetime('2020-02-28')) ]

    # Sort the DataFrame by date
    df_end_of_month = df_end_of_month.sort_values('date')
    
    # Prepare the data for the stream graph
    data = []
    for country in df_end_of_month['location'].unique():
        country_data = df_end_of_month[df_end_of_month['location'] == country]
        trace = go.Stream(
            x=country_data['month'],
            y=country_data['total_cases'],
            name=country,
            hovertemplate='Month: %{x}<br>Total Cases: %{y}',
            mode='lines',
            stackgroup='one'
        )
        data.append(trace)

    # Create the stream graph layout
    layout = go.Layout(
        title='Total Monthly COVID-19 Cases by Country',
        xaxis=dict(title='Months'),
        yaxis=dict(title='Total Number of COVID-19 Cases'),
        hovermode='closest'
    )

    # Create the stream graph figure
    fig = go.Figure(data=data, layout=layout)

    # Convert the figure to JSON for compatibility
    graph1JSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    #################################################### Graph 2 ####################################################

    

    # Define the indices
    us_indices = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow Jones, NASDAQ
    eu_indices = ['^FTSE', '^GDAXI', '^FCHI']  # FTSE 100, DAX, CAC 40
    asia_indices = ['^N225', '^HSI', '000001.SS']  # Nikkei 225, Hang Seng, Shanghai

    all_indices = us_indices + eu_indices + asia_indices

    # Fetch the data
    # data = yf.download(all_indices, start="2019-01-01", end="2023-11-14")

    # Save to CSV
    # data.to_csv('data.csv')

    # Read data from CSV
    data = pd.read_csv('application\data.csv', header=[0,1], index_col=0, parse_dates=True).loc['2019-08-01':'2023-11-14']

    # We are interested in adjusted closing prices
    data = data['Adj Close']

    # Compute Cumulative Returns
    returns = (1+data.pct_change()).cumprod()

    # Take average of each region (US, EU, Asia)
    returns['US'] = returns[us_indices].mean(axis=1)
    returns['EU'] = returns[eu_indices].mean(axis=1)
    returns['Asia'] = returns[asia_indices].mean(axis=1)

    # Remove individual indices
    returns = returns.drop(us_indices, axis=1)
    returns = returns.drop(eu_indices, axis=1)
    returns = returns.drop(asia_indices, axis=1)

    # Define periods
    periods = ['Pre-COVID Growth', 'COVID Crisis', 'Post-Crisis Recovery', 'Pandemic Aftermath']
    dates = [pd.Timestamp('2019-08-01'), pd.Timestamp('2020-02-01'), pd.Timestamp('2020-04-30'), 
            pd.Timestamp('2021-12-31'), pd.Timestamp('2023-11-14')]

    # Create a new column for the period
    returns['Period'] = pd.cut(returns.index, bins=dates, labels=periods, right=False)

    # Melt the data for the facet grid
    melted_returns = returns.reset_index().melt(id_vars=['Date', 'Period'], 
                                                var_name='Index', value_name='Cumulative Return').dropna()

    # Create figure and axis
    fig, ax = plt.subplots()

    # Initialization function for the start of the animation
    def init():
        ax.clear()

    # Update function for each frame of the animation
    def update(frame):
        ax.clear()
        current_date = (returns.index[0] + pd.DateOffset(days=frame)).date()
        temp_df = melted_returns[(current_date <= melted_returns['Date'].dt.date) & (melted_returns['Date'].dt.date <= current_date + pd.DateOffset(days=frame))]
        if not temp_df.empty:
            sns.violinplot(x="Index", y="Cumulative Return", data=temp_df, ax=ax)
            ax.set_title(f'Stock Market at {current_date}')
            period = temp_df['Period'].unique()[0]  # Get the period
            ax.annotate(f'Period: {period}', xy=(0.5, 0.9), xycoords='axes fraction')  # Display the period

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=range(0, (returns.index[-1] - returns.index[0]).days, 5), init_func=init)


    # Saving the video 

    # ani.save('violin_plot.mp4', dpi=300)
    # Display the animation
    # HTML(ani.to_html5_video())
    
    video_url_violin = '/static/violin_plot.mp4'
    
    
    ################################################# Graph 3 #################################################
    # Define the indices
    us_indices = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow Jones, NASDAQ
    eu_indices = ['^FTSE', '^GDAXI', '^FCHI']  # FTSE 100, DAX, CAC 40
    asia_indices = ['^N225', '^HSI', '000001.SS']  # Nikkei 225, Hang Seng, Shanghai


    all_indices = us_indices + eu_indices + asia_indices

    # Fetch the data
    # data = yf.download(all_indices, start="2019-01-01", end="2023-11-14")

    # # save to csv
    # data.to_csv('data.csv')

    # read data from csv
    data = pd.read_csv('application\data.csv', header=[0,1], index_col=0, parse_dates=True).loc['2019-08-01':'2022-02-28']

    covid_data = pd.read_csv('application\Cleaned Covid Data.csv', parse_dates=['date'])
    usa_covid_data = covid_data[covid_data['location'] == 'United States']
    china_covid_data = covid_data[covid_data['location'] == 'China']
    europe_covid_data = covid_data[covid_data['continent'] == 'Europe']

    usa_total_cases = usa_covid_data.set_index('date')['total_cases']
    china_total_cases = china_covid_data.set_index('date')['total_cases']
    europe_total_cases = europe_covid_data.groupby('date').sum()['total_cases']

    # We are interested in adjusted closing prices
    data = data['Adj Close']

    # Compute Cumulative Returns
    returns = (1+data.pct_change()).cumprod()

    usa_returns = (data['^GSPC'].pct_change())
    china_returns = (data['000001.SS'].pct_change())
    europe_returns = (data['^FTSE'].pct_change())

    # merge usa_return with total cases , right join
    usa = pd.merge(usa_returns, usa_total_cases, how='inner', left_index=True, right_index=True)
    usa['total_cases'] = usa['total_cases'].pct_change()
    usa = usa.dropna()
    usa_rolling_correlation = usa[['total_cases','^GSPC']].rolling(30).corr().dropna().loc[pd.IndexSlice[:, '^GSPC'], 'total_cases']
    usa_rolling_correlation.index = usa_rolling_correlation.index.droplevel(1)
    usa_rolling_correlation.name = 'usa'

    # do the same for china and europe
    china = pd.merge(china_returns, china_total_cases, how='inner', left_index=True, right_index=True)
    china['total_cases'] = china['total_cases'].pct_change()
    china = china.dropna()
    china_rolling_correlation = china[['total_cases','000001.SS']].rolling(30).corr().dropna().loc[pd.IndexSlice[:, '000001.SS'], 'total_cases']
    china_rolling_correlation.index = china_rolling_correlation.index.droplevel(1)
    china_rolling_correlation.name = 'china'

    europe = pd.merge(europe_returns, europe_total_cases, how='inner', left_index=True, right_index=True)
    europe['total_cases'] = europe['total_cases'].pct_change()
    europe = europe.dropna()
    europe_rolling_correlation = europe[['total_cases','^FTSE']].rolling(30).corr().dropna().loc[pd.IndexSlice[:, '^FTSE'], 'total_cases']
    europe_rolling_correlation.index = europe_rolling_correlation.index.droplevel(1)
    europe_rolling_correlation.name = 'europe'

    # merge all rolling correlation, inner join
    rolling_correlation = pd.merge(usa_rolling_correlation, china_rolling_correlation, how='inner', left_index=True, right_index=True)
    rolling_correlation = pd.merge(rolling_correlation, europe_rolling_correlation, how='inner', left_index=True, right_index=True)

    # take average of each region (us, eu, asia)
    returns['US'] = returns[us_indices].mean(axis=1)
    returns['EU'] = returns[eu_indices].mean(axis=1)
    returns['Asia'] = returns[asia_indices].mean(axis=1)

    # remove individual indices
    returns = returns.drop(us_indices, axis=1)
    returns = returns.drop(eu_indices, axis=1)
    returns = returns.drop(asia_indices, axis=1)

    # Define periods
    periods = ['Pre-COVID', 'Outbreak in China', 'Global Spread', 'Lockdowns Begin', 
            'Peak First Wave', 'Initial Reopenings', 'Second Wave', 'Vaccine Development', 
            'Vaccine Approval', 'Vaccine Rollout Begins', 'Second Wave Peaks', 'Lockdowns Ease', 
            'Vaccine Distribution Widens', 'New Variants Emerge', 'Third Wave', 'Global Vaccination Efforts', 
            'Vaccine Hesitancy', 'Variants of Concern', 'Booster Vaccines', 'Endemic Transition']
    dates = [pd.Timestamp('2019-08-01'), pd.Timestamp('2019-11-17'), pd.Timestamp('2020-01-31'), 
            pd.Timestamp('2020-03-11'), pd.Timestamp('2020-05-01'), pd.Timestamp('2020-07-01'), 
            pd.Timestamp('2020-09-01'), pd.Timestamp('2020-11-01'), pd.Timestamp('2020-12-09'), 
            pd.Timestamp('2021-01-01'), pd.Timestamp('2021-03-01'), pd.Timestamp('2021-05-01'), 
            pd.Timestamp('2021-07-01'), pd.Timestamp('2021-09-01'), pd.Timestamp('2021-11-01'), 
            pd.Timestamp('2022-01-01'), pd.Timestamp('2022-03-01'), pd.Timestamp('2022-05-01'), 
            pd.Timestamp('2022-07-01'), pd.Timestamp('2022-09-01'), pd.Timestamp('2023-12-31')]

    # Create a new column for the period
    returns['Period'] = pd.cut(returns.index, bins=dates, labels=periods, right=False)

    # Melt the data for the facet grid
    melted_returns = returns.reset_index().melt(id_vars=['Date', 'Period'], 
                                                var_name='Index', value_name='Cumulative Return').dropna()

    

    
    

    # Create figure and axis
    fig, ax = plt.subplots()


   # Create the dictionary
    period_dict = {dates[i]: periods[i] for i in range(len(dates)-1)}

    # Map the period names to the dates in the dataframe
    rolling_correlation['period'] = pd.cut(rolling_correlation.index, 
                                        bins=dates, 
                                        labels=periods, 
                                        right=False)

    #print(rolling_correlation)


    # Set the style of seaborn
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(15, 6))

    line_usa, = ax.plot([], [], label='USA',alpha=1)
    line_china, = ax.plot([], [], label='China',alpha=0.5)
    line_europe, = ax.plot([], [], label='Europe',alpha=0.5)

    ax.legend()

    def init():
        ax.set_xlim(rolling_correlation.index.min(), rolling_correlation.index.max())
        ax.set_ylim(rolling_correlation.drop(columns='period').min().min(), rolling_correlation.drop(columns='period').max().max())
        
        # Remove the existing title
        ax.set_title('Correlation Between Covid Cases Pct Changes and Returns')
        
        return line_usa, line_china, line_europe

    def update(i):
        x = rolling_correlation.index[:i+1]
        line_usa.set_data(x, rolling_correlation['usa'].iloc[:i+1])
        line_china.set_data(x, rolling_correlation['china'].iloc[:i+1])
        line_europe.set_data(x, rolling_correlation['europe'].iloc[:i+1])
        
        # Update the period name in the title
        period = rolling_correlation['period'].iloc[i]
        ax.set_title(f"Correlation Between Covid Cases Pct Changes and Returns - Period: {period}")
        
        return line_usa, line_china, line_europe

    ani = animation.FuncAnimation(fig, update, frames=range(len(rolling_correlation)), init_func=init, blit=True)
    
    video_url_correlation= '/static/correlation_animation_final.mp4'
    
    
     ################################################# Graph 4 #################################################
     
    df=pd.read_csv('application\Inflation_transposed.csv',header=0)
    df.set_index('Date ',inplace=True)
    # Convert the index to datetime format
    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True,inplace=True)
    
    
    # bcr.bar_chart_race(
    # df=df,
    # orientation='h',
    # filename='inflation_new.mp4',
    # sort='desc',
    # n_bars=11,
    # fixed_order=False,
    # fixed_max=False,
    # steps_per_period=10,
    # period_length=500,
    # end_period_pause=0,
    # interpolate_period=False,
    # period_label={'x': .98, 'y': .3, 'ha': 'right', 'va': 'center'},
    # period_template='%B %d, %Y',

    # perpendicular_bar_func='median',
    # colors='dark12',
    # title='Inflation rate by Country',
    # bar_size=.95,
    # bar_textposition='inside',
    # bar_texttemplate='{x:,.0f}',
    # bar_label_font=7,
    # tick_label_font=7,
    # tick_template='{x:,.0f}',
    # shared_fontdict=None,
    # scale='linear',
    # fig=None,
    # writer=None,
    # bar_kwargs={'alpha': .7},
    # fig_kwargs={'figsize': (6, 3.5), 'dpi': 144},
    # filter_column_colors=False)
    
    
    video_url_inflation='/static/inflation_final.mp4'

    
        
    
        
    
    
    
    
    return render_template("index.html",title="Home",graph1JSON=graph1JSON,
                           video_url_violin=video_url_violin,
                           video_url_correlation=video_url_correlation,
                           video_url_inflation=video_url_inflation
                           )
    
    
# dash_app = create_dash_application(flask_app)
# server=dash_app.server
# dash_app.title = 'My Dash App' 
   
@flask_app.route("/dash1/")
def dash1():
    # Make a GET request to the Dash app URL
    response = requests.get('http://localhost:5000/dash1/')

    # Access the HTML content
    html_content = response.text

    # Modify the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the body element
    body = soup.find('body')

    # Create the navbar element
    navbar = soup.new_tag('nav')
    navbar['class'] = 'navbar'
    navbar_content = ''' <a class="navbar-brand" href="#">COMP 4462</a>
          <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarText" aria-controls="navbarText" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
          </button>
          <div class="collapse navbar-collapse" id="navbarText">
            <ul class="navbar-nav ml-auto mb-2 mb-lg-0">
                  <li class="nav-item">
                    <a class="nav-link" href="/">Home</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="/dash1/">Stock Indices</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="/dash2/">Bivariate Map</a>
                </li>
            </ul>'''
    navbar.append(BeautifulSoup(navbar_content, 'html.parser'))

    # Append the navbar to the body
    body.insert(0, navbar)

    # Get the modified HTML content
    modified_html_content = str(soup)

    return modified_html_content


@flask_app.route("/dash2/")
def dash2():
    return render_template('application/templates/dash2.html',title="Dash2")

# @flask_app.route('/static/main.css')
# def serve_css():
#     return flask_app.send_static_file('main.css')




