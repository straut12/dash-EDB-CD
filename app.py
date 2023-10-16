import pandas as pd
import matplotlib.pyplot as plt

import dash                               # installed odfpy for excel/ods reading
import dash_bootstrap_components as dbc  # installed dash_bootstrap_templates too
from dash import dcc, html
from dash import dash_table as dt
import plotly.express as px
from dash.dependencies import Input, Output

# Dash apps are Flask apps

# https://dash.plotly.com/tutorial
# https://bootswatch.com/
# https://hellodash.pythonanywhere.com/
# https://hellodash.pythonanywhere.com/adding-themes/datatable
# https://community.plotly.com/t/styling-dash-datatable-select-rows-radio-button-and-checkbox/59466/3

#============IMPORT DATA================
# Get offline data for box plot comparison
#df = pd.read_excel('dht11-temp-data2.ods', engine='odf')
# Manually created date column in spreadsheet and truncated to day only (no H:M). 
# Tried pd.to_datetime(df['_time'], format="%Y-%m-%d").dt.floor("d") but it left .0000000 for the H:M. May have been ok.
# DatetimeProperties.to_pydatetime is deprecated, in future version will return a Series containing python datetime objects instead of an ndarray. To retain the old behavior, call `np.array` on the result
df = pd.read_csv('dht11-temp-data').assign(date=lambda data: pd.to_datetime(data["date"], format="%Y-%m-%d")) # was not uploading with csv extension
df['location'] = df['location'].astype('str') # chart was not working because this was read as an int. had to convert to str
# DO NOT USE 1,2,3 for labels. Can have confusing results due to int vs str scenarios

# +00:00 is Hour Min offset from UTC
# the freq/period parameter in pandas.date_range refers to the interval between dates generated. The default is "D", but it could be hourly, monthly, or yearly. 

#df['_time'] = pd.to_datetime(df['date'], unit='d', origin='1899-12-30') # Changed the decimal by a little and removed the +00:00
df['_time'] = pd.to_datetime((df['_time'])) # converted _time from obj to datetime64 with tz=UTC

#for x in ["date"]:    # another method
#    if x in df.columns:
#        df[x] = pd.to_datetime(df['_time'], format="%Y-%m-%d").dt.floor("d")

#==CREATE TABLES/GRAPHS THAT ARE NOT CREATED WITH CALLBACK (not interactive)=====
# Create summary dataframe with statistics
dfsummary = df.groupby('location')['tempf'].describe()  # describe outputs a dataframe
dfsummary = dfsummary.reset_index()  # this moves the index (locations 1,2,3,4) into a regular column so they show up in the dash table
'''dfsummary.style.format({   # this would work if the values were floats. However they
    "mean": "{:.1f}",         # were strings after the describe functions so had to use
    "std": "{:.1f}",          # the map function below
})'''
dfsummary.loc[:, "mean"] = dfsummary["mean"].map('{:.1f}'.format)
dfsummary.loc[:, "std"] = dfsummary["std"].map('{:.1f}'.format)
dfsummary.loc[:, "50%"] = dfsummary["50%"].map('{:.1f}'.format)
table = dbc.Table.from_dataframe(dfsummary, striped=True, bordered=True, hover=True)

histogram1 = px.histogram(df, x="tempf", nbins=30)

#===START DASH AND CREATE LAYOUT OF TABLES/GRAPHS===========
# Use dash bootstrap components (dbc) for styling
dbc_css = "assets/dbc.css"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR, dbc_css])
# available themes: BOOTSTRAP, CERULEAN, COSMO, CYBORG, DARKLY, FLATLY, JOURNAL, LITERA, LUMEN, LUX, MATERIA, MINTY, MORPH, PULSE, QUARTZ, SANDSTONE, SIMPLEX, SKETCHY, SLATE, SOLAR, SPACELAB, SUPERHERO, UNITED, VAPOR, YETI, ZEPHYR

# Layout of the dash graphs, tables, drop down menus, etc
# Using dbc container for styling/formatting
app.layout = dbc.Container(html.Div([
    html.Div(["Home Temp Data from DHT11 (units are F)",table], style={'display': 'inline-block', 'width': '50%'}),
    html.Div(["Date Range",
    dcc.DatePickerRange(
        id="date-range",
        min_date_allowed=df["date"].min().date(),
        max_date_allowed=df["date"].max().date(),
        start_date=df["date"].min().date(),
        end_date=df["date"].max().date(),
    )], style={'display': 'inline-block', 'width': '50%'}),
    html.Div('Sensor location 1:IndoorA 2:Basement 3:IndoorB 4:Outdoors'),
    dcc.Checklist(
        id="checklist",  # id names will be used by the callback to identify the components
        options=["1", "2", "3","4"],
        value=["1", "2", "3", "4"], # default selections
        inline=True),
    html.Div([dcc.Graph(figure={}, id='linechart1')], style={'display': 'inline-block', 'width': '50%'}),  # figure is blank dict because created in callback below
    html.Div([dcc.Graph(figure=histogram1, id='hist1')], style={'display': 'inline-block', 'width': '50%'}),
    html.Div([
    html.P("y-axis:"),
    dcc.RadioItems(
        id='y-axis', 
        options=[{'value': x, 'label': x} 
                 for x in ['humidityi', 'tempf']],
        value='tempf', 
        labelStyle={'display': 'inline-block'}
    ),
    html.Div([dcc.Graph(figure={}, id="box-plot1")], style={'display': 'inline-block', 'width': '50%'}),
    html.Div([dcc.Graph(figure={}, id="box-plot2")], style={'display': 'inline-block', 'width': '50%'}),
    ])
]), fluid=True, className="dbc dbc-row-selectable")

#=====CREATE INTERACTIVE GRAPHS=============
@app.callback(
    Output("linechart1", "figure"),    # args are component id and then component property
    Input("checklist", "value"),        # args are component id and then component property. component property is passed
    Input("date-range", "start_date"),  # in order to the chart function below
    Input("date-range", "end_date"))
def update_line_chart(sensor, start_date, end_date):    # callback function arg 'sensor' refers to the component property of the input or "value" above
    filtered_data = df.query("date >= @start_date and date <= @end_date")
    mask = filtered_data.location.isin(sensor)
    fig = px.line(filtered_data[mask], 
        x='_time', y='tempf', color='location')
    return fig

@app.callback(
    Output("box-plot1", "figure"), 
    Input("y-axis", "value"))
def generate_chart(y):
    fig = px.box(df, x="location", y=y, color="ventilator")
    return fig

@app.callback(
    Output("box-plot2", "figure"), 
    Input("y-axis", "value"))
def generate_chart(y):
    fig = px.box(df, x="location", y=y, color="Outside-humidity")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
