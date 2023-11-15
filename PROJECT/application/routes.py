from application import app 
from flask import render_template,url_for
import pandas as pd 
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


@app.route("/")
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
        
    
    

   
    
    return render_template("index.html",title="Home",graph1JSON=graph1JSON)