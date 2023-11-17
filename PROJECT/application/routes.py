from application import app 
from flask import render_template,url_for
import pandas as pd 
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
    
    
    ## Graph 2
    
    

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
    
    video_url = '/static/violin_plot.mp4'
    
    
        
    
    

   
    
    return render_template("index.html",title="Home",graph1JSON=graph1JSON,video_url=video_url)