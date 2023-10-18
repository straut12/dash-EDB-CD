import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

                              # installed odfpy for excel/ods reading
import dash_bootstrap_components as dbc  # installed dash_bootstrap_templates too
from dash import Dash, dcc, html
from dash import dash_table as dt
import plotly.express as px
import plotly.graph_objects as go
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
df = pd.read_csv('CD-ALL-synthetic-data').assign(date=lambda data: pd.to_datetime(data["Date"], format="%Y-%m-%d")) # was not uploading with csv extension

# df['Lot'] = df['Lot'].astype('str') charts were not working because this was read as an int. had to convert to str
for name, values in df.iloc[:, 2:14].items():
    df[name] = df[name].astype(str)

for name, values in df.iloc[:, 20:21].items():
    df[name] = df[name].round(2)

target = df['Target'].iloc[0] # get initial target for default wafer maps
dflt_specl = [df['LS'].iloc[0], df['US'].iloc[0]] # get initial spec limits for default wafer maps
lotdflt = df['Lot'].iloc[0] # get initial lot,wfr for default wafer maps
wfrdflt = df['Wfr'].iloc[0]
tooll = np.sort(df['Tool'].unique()).tolist() # get a unique list of tools for the graph selection menu and colors
lotl = np.sort(df['Lot'].unique()).tolist()
wfrl = np.sort(df['Wfr'].unique()).tolist()
# DO NOT USE 1,2,3 for labels. Can have confusing results due to int vs str scenarios

# +00:00 is Hour Min offset from UTC
# the freq/period parameter in pandas.date_range refers to the interval between dates generated. The default is "D", but it could be hourly, monthly, or yearly. 

#df['_time'] = pd.to_datetime(df['date'], unit='d', origin='1899-12-30') # Changed the decimal by a little and removed the +00:00
df['DateTime'] = pd.to_datetime((df['DateTime'])) # converted _time from obj to datetime64 with tz=UTC

#for x in ["date"]:    # another method
#    if x in df.columns:
#        df[x] = pd.to_datetime(df['_time'], format="%Y-%m-%d").dt.floor("d")

#==CREATE TABLES/GRAPHS THAT ARE NOT CREATED WITH CALLBACK (not interactive)=====
# Create summary dataframe with statistics
dfsummary = df.groupby('Tool')['MP1'].describe()  # describe outputs a dataframe
dfsummary = dfsummary.reset_index()  # this moves the index (locations 1,2,3,4) into a regular column so they show up in the dash table
'''dfsummary.style.format({   # this would work if the values were floats. However they
    "mean": "{:.1f}",         # were strings after the describe functions so had to use
    "std": "{:.1f}",          # the map function below
})'''
dfsummary.loc[:, "mean"] = dfsummary["mean"].map('{:.1f}'.format)
dfsummary.loc[:, "std"] = dfsummary["std"].map('{:.1f}'.format)
dfsummary.loc[:, "50%"] = dfsummary["50%"].map('{:.1f}'.format)
table = dbc.Table.from_dataframe(dfsummary, striped=True, bordered=True, hover=True)

#histogram1 = px.histogram(df, x="CD", nbins=30)

#===START DASH AND CREATE LAYOUT OF TABLES/GRAPHS===========
# Use dash bootstrap components (dbc) for styling
dbc_css = "assets/dbc.css"

app = Dash(__name__, external_stylesheets=[dbc.themes.SOLAR, dbc_css])
# available themes: BOOTSTRAP, CERULEAN, COSMO, CYBORG, DARKLY, FLATLY, JOURNAL, LITERA, LUMEN, LUX, MATERIA, MINTY, MORPH, PULSE, QUARTZ, SANDSTONE, SIMPLEX, SKETCHY, SLATE, SOLAR, SPACELAB, SUPERHERO, UNITED, VAPOR, YETI, ZEPHYR

# Layout of the dash graphs, tables, drop down menus, etc
# Using dbc container for styling/formatting
app.layout = dbc.Container(html.Div([
    html.Div(["Date Range",
        dcc.DatePickerRange(
            id="date-range",
            min_date_allowed=df["date"].min().date(),
            max_date_allowed=df["date"].max().date(),
            start_date=df["date"].min().date(),
            end_date=df["date"].max().date(),
        )], style={'display': 'inline-block', 'width': '50%'}),
    html.Div(["CD (nm)",table], style={'display': 'inline-block', 'width': '50%'}),
    html.Div([dcc.RangeSlider(55, 75, 0.5, value=dflt_specl, tooltip={"placement": "bottom", "always_visible": False}, id='limit-slider')], style={'display': 'inline-block', 'width': '100%'}),
    html.Div('Tools', style={'display': 'inline-block', 'width': '10%'}),
    html.Div(
    dcc.RadioItems(
        id='chart-y', 
        options=[{'value': x, 'label': x}  # radio button labels and values
                 for x in ['MP1', 'MP2']],  # radio button labels and values
        value='MP1',   # Default
        labelStyle={'display': 'inline-block'}
        ), style={'display': 'inline-block', 'width': '40%'}),
    html.Div(
        dcc.RadioItems(
        id='boxplt-y', 
        options=[{'value': x, 'label': x}  # radio button labels and values
                 for x in ['MP1', 'MP2']],  # radio button labels and values
        value='MP1',   # Default
        labelStyle={'display': 'inline-block'}
        ), style={'display': 'inline-block', 'width': '50%'}),
    html.Div(dcc.Checklist(
        id="tool_list",  # id names will be used by the callback to identify the components
        options=tooll, # list of the tools
        value=tooll, # default selections
        inline=True), style={'display': 'inline-block', 'width': '50%'}),
    html.Div(
        dcc.RadioItems(
        id='unit', 
        options=[{'value': x, 'label': x}  # radio button labels and values
                for x in ['iARC', 'COAT', 'CHUCK', 'PEB', 'DVLP']],  # radio button labels and values
        value='PEB',   # Default
        labelStyle={'display': 'inline-block'}
        ), style={'display': 'inline-block', 'width': '50%'}),
    html.Div([dcc.Graph(figure={}, id='linechart1')], style={'display': 'inline-block'}),  # figure is blank dict because created in callback below
    html.Div([dcc.Graph(figure={}, id='box-plot1')], style={'display': 'inline-block'}),
    html.Br(style={"line-height": "5"}),
    html.Div(
    dcc.RadioItems(
        id='cntr1-radio', 
        options=[{'value': x, 'label': x}  # radio button labels and values
                 for x in ['Auto', 'Manual']],  # radio button labels and values
        value='Auto',   # Default
        labelStyle={'display': 'inline-block'}
        ), style={'display': 'inline-block', 'width': 430}),
    html.Div(
    dcc.RadioItems(
        id='cntr2-radio', 
        options=[{'value': x, 'label': x}  # radio button labels and values
                 for x in ['Auto', 'Manual']],  # radio button labels and values
        value='Auto',   # Default
        labelStyle={'display': 'inline-block'}
        ), style={'display': 'inline-block', 'width': 430}),
    html.Br(style={"line-height": "5"}),
    html.Div([dcc.RangeSlider(60, 70, 0.5, value=[60, 70], tooltip={"placement": "bottom", "always_visible": False}, id='cntr1-slider')], style={'display': 'inline-block', 'width': 430}),
    html.Div([dcc.RangeSlider(60, 70, 0.5, value=[60, 70], tooltip={"placement": "bottom", "always_visible": False}, id='cntr2-slider')], style={'display': 'inline-block', 'width': 430}),
    html.Br(style={"line-height": "5"}),
    html.Div([dcc.Dropdown(lotl, lotdflt, id='lot1-dd')], style={'display': 'inline-block', 'width': 215}),
    html.Div([dcc.Dropdown(wfrl, wfrdflt, id='wfr1-dd')], style={'display': 'inline-block', 'width': 215}),
    html.Div([dcc.Dropdown(lotl, lotdflt, id='lot2-dd')], style={'display': 'inline-block', 'width': 215}),
    html.Div([dcc.Dropdown(wfrl, wfrdflt, id='wfr2-dd')], style={'display': 'inline-block', 'width': 215}),
    html.Br(style={"line-height": "5"}),
    html.Div([dcc.Graph(figure={}, id='cntr1')], style={'display': 'inline-block', 'width': 430}),
    html.Div([dcc.Graph(figure={}, id='cntr2')], style={'display': 'inline-block', 'width': 430}),
]), fluid=True, className="dbc dbc-row-selectable")

#=====CREATE INTERACTIVE GRAPHS=============
# Create line chart
@app.callback(
    Output("linechart1", "figure"),    # args are component id and then component property
    Input("tool_list", "value"),        # args are component id and then component property. component property is passed
    Input("date-range", "start_date"),  # in order to the chart function below
    Input("date-range", "end_date"),
    Input("chart-y", "value"),
    Input("limit-slider", "value"))
def update_line_chart(tool, start_date, end_date, y, limits):    # callback function arg 'tool' refers to the component property of the input or "value" above
    filtered_data = df.query("date >= @start_date and date <= @end_date")  # Get only data within time frame selected
    mask = filtered_data.Tool.isin(tool)                                   # Create a panda series with True/False of only tools selected 
    fig = px.line(filtered_data[mask], 
        x='DateTime', y=y, color='Tool'
        ,category_orders={'Tool':tooll}  # can manually set colors color_discrete_sequence = ['darkred', 'dodgerblue', 'green', 'tan']
        ,color_discrete_sequence = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
        ,line_shape="hv"
        ,hover_data=['Lot', 'Wfr', 'Slot', 'iARC', 'COAT', 'CHUCK', 'PEB', 'DVLP', 'Site', 'Xmm', 'Ymm']
        ,markers=True)
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Create box plot
@app.callback(
    Output("box-plot1", "figure"), 
    Input("boxplt-y", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("unit", "value"),
    Input("limit-slider", "value"))
def generate_chart(y, start_date, end_date, unit, limits):
    filtered_data = df.query("date >= @start_date and date <= @end_date")
    fig = px.box(filtered_data, x="Tool", y=y, color=unit)
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

@app.callback(
    Output("cntr1", "figure"), 
    Input("lot1-dd", "value"),
    Input("wfr1-dd", "value"),
    Input("cntr1-radio", "value"),
    Input("cntr1-slider", "value"))
def generate_chart(lotID, wfrID, radio, cntr_limits):
    dfcntr = df.loc[(df['Lot'] == lotID) & (df['Wfr'] == wfrID )]
    dfcntr = dfcntr.drop(['Date', 'date', 'LGPT', 'iARC', 'COAT', 'CHUCK', 'PEB','DVLP','Target','LS','US'], axis=1)
    # Create model to predict MP1 where there was no measurement
    features = dfcntr.iloc[:, -4:-2].values
    label = dfcntr.iloc[:, -1].values
    features_train, features_test, label_train, label_test = train_test_split(features, label, test_size = 0.1, random_state=0)
    poly_reg = PolynomialFeatures(degree = 4)
    X_poly = poly_reg.fit_transform(features_train)
    regressor = LinearRegression()
    regressor.fit(X_poly, label_train)
    # Create 2D matrix: X, Y, Z for contour plot
    xcoord = np.append(dfcntr.iloc[:, -4:-3].values, [i for i in range(-150, 160, 10)]) # create dummy X/Y on the edge and append to the xmm/ymm lists for better edge coverage of predictions
    ycoord = np.append(dfcntr.iloc[:, -3:-2].values, [i for i in range(-150, 160, 10)])
    xmm = np.sort(np.unique(xcoord))
    ymm = np.sort(np.unique(ycoord))
    #xmm = np.sort(np.unique(dfcntr.iloc[:, -4:-3].values)) # get the unique Xmm location values
    #ymm = np.sort(np.unique(dfcntr.iloc[:, -3:-2].values)) # get the unique Ymm location values
    xplt = np.array(xmm).tolist()
    yplt = np.array(ymm).tolist()
    X,Y = np.meshgrid(xmm, ymm)
    Z = np.zeros((X.shape))
    
    # Create a dict that maps all the measured MP1s with their X,Y loc
    dict = dfcntr.to_dict('list')
    MP1_map={}
    for i in range(len(dict['MP1'])):
        MP1_map[str(dict['Xmm'][i]) + '-' + str(dict['Ymm'][i])] = dict['MP1'][i]
    # Create a full 2D map of MP1 for every X/Y loc. Use meas values if present, otherwise fill in with predicted values
    rmax = dfcntr['Rmm'].max()    # will only predict values beyond the radius of measured values. will let plotly contour fill in the middle of the wafer
    for i in range(Z.shape[1]):
        for j in range(Z.shape[0]):
            if MP1_map.get(str(X[0][i]) + '-' + str(Y[j][0])) == None:
                radius = np.sqrt(X[0][i]**2 + Y[j][0]**2)
                if radius > rmax and radius < 150:    # will only predict values beyond the radius of measured values.
                    pred_value = regressor.predict(poly_reg.transform([[X[0][i], Y[j][0]]]))
                    Z[j][i] = pred_value[0]
            else:
                Z[j][i] = MP1_map[str(X[0][i]) + '-' + str(Y[j][0])]    # point was measured so fill in with measured value
    Z = np.where(Z==0, np.nan, Z) # replace 0's with nan
    #Zdf = dfcntr.drop(['DateTime', 'Lot', 'Wfr', 'Slot', 'Tool', 'MP', 'Site'], axis=1)
    #Zarray = Zdf.pivot(index="Xmm", columns="Ymm", values="MP1").to_numpy()
    if radio =="Auto":
        contoursd = {'coloring':'heatmap', 'showlabels':True, 'labelfont':{'size':12, 'color':'white'}} # can add 'start':64, 'end':67, size=4
    else:
        contoursd = {'coloring':'heatmap', 'start':cntr_limits[0], 'end':cntr_limits[1], 'showlabels':True, 'labelfont':{'size':12, 'color':'white'}} # can add 'start':64, 'end':67, size=4
    fig = go.Figure(data=
        go.Contour(
            z=Z,
            x=xmm,
            y=ymm,
            colorscale='Turbo',
            connectgaps=True,
            contours=contoursd,
            colorbar={'title': dfcntr['Tool'].iloc[0]}
            ))
    title = str(dfcntr['DateTime'].iloc[0])[:19]
    fig.update_layout(title={'text': title, 'font': {'size': 12}})
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=-150, y0=-150,
        x1=150, y1=150,
        opacity=0.2,
        fillcolor="blue",
        line_color="black",
        ) 
    return fig

@app.callback(
    Output("cntr2", "figure"), 
    Input("lot2-dd", "value"),
    Input("wfr2-dd", "value"),
    Input("cntr2-radio", "value"),
    Input("cntr2-slider", "value"))
def generate_chart(lotID, wfrID, radio, cntr_limits):
    dfcntr = df.loc[(df['Lot'] == lotID) & (df['Wfr'] == wfrID )]
    dfcntr = dfcntr.drop(['Date', 'date', 'LGPT', 'iARC', 'COAT', 'CHUCK', 'PEB','DVLP','Target','LS','US'], axis=1)
    # Create model to predict MP1 where there was no measurement
    features = dfcntr.iloc[:, -4:-2].values
    label = dfcntr.iloc[:, -1].values
    features_train, features_test, label_train, label_test = train_test_split(features, label, test_size = 0.1, random_state=0)
    poly_reg = PolynomialFeatures(degree = 4)
    X_poly = poly_reg.fit_transform(features_train)
    regressor = LinearRegression()
    regressor.fit(X_poly, label_train)
    # Predicting the Test set results
    label_pred = regressor.predict(poly_reg.transform(features_test))
    r2 = round(r2_score(label_test, label_pred),0)
    # Create 2D matrix: X, Y, Z for contour plot
    xcoord = np.append(dfcntr.iloc[:, -4:-3].values, [i for i in range(-150, 160, 10)]) # create dummy X/Y on the edge and append to the xmm/ymm lists for better edge coverage of predictions
    ycoord = np.append(dfcntr.iloc[:, -3:-2].values, [i for i in range(-150, 160, 10)])
    xmm = np.sort(np.unique(xcoord))
    ymm = np.sort(np.unique(ycoord))
    #xmm = np.sort(np.unique(dfcntr.iloc[:, -4:-3].values)) # get the unique Xmm location values
    #ymm = np.sort(np.unique(dfcntr.iloc[:, -3:-2].values)) # get the unique Ymm location values
    xplt = np.array(xmm).tolist()
    yplt = np.array(ymm).tolist()
    X,Y = np.meshgrid(xmm, ymm)
    Z = np.zeros((X.shape))
    
    # Create a dict that maps all the measured MP1s with their X,Y loc
    dict = dfcntr.to_dict('list')
    MP1_map={}
    for i in range(len(dict['MP1'])):
        MP1_map[str(dict['Xmm'][i]) + '-' + str(dict['Ymm'][i])] = dict['MP1'][i]
    # Create a full 2D map of MP1 for every X/Y loc. Use meas values if present, otherwise fill in with predicted values
    rmax = dfcntr['Rmm'].max()    # will only predict values beyond the radius of measured values. will let plotly contour fill in the middle of the wafer
    for i in range(Z.shape[1]):
        for j in range(Z.shape[0]):
            if MP1_map.get(str(X[0][i]) + '-' + str(Y[j][0])) == None:
                radius = np.sqrt(X[0][i]**2 + Y[j][0]**2)
                if radius > rmax and radius < 150:    # will only predict values beyond the radius of measured values.
                    pred_value = regressor.predict(poly_reg.transform([[X[0][i], Y[j][0]]]))
                    Z[j][i] = pred_value[0]
            else:
                Z[j][i] = MP1_map[str(X[0][i]) + '-' + str(Y[j][0])]    # point was measured so fill in with measured value
    Z = np.where(Z==0, np.nan, Z) # replace 0's with nan
    #Zdf = dfcntr.drop(['DateTime', 'Lot', 'Wfr', 'Slot', 'Tool', 'MP', 'Site'], axis=1)
    #Zarray = Zdf.pivot(index="Xmm", columns="Ymm", values="MP1").to_numpy()
    if radio =="Auto":
        contoursd = {'coloring':'heatmap', 'showlabels':True, 'labelfont':{'size':12, 'color':'white'}} # can add 'start':64, 'end':67, size=4
    else:
        contoursd = {'coloring':'heatmap', 'start':cntr_limits[0], 'end':cntr_limits[1], 'showlabels':True, 'labelfont':{'size':12, 'color':'white'}} # can add 'start':64, 'end':67, size=4
    fig = go.Figure(data=
        go.Contour(
            z=Z,
            x=xmm,
            y=ymm,
            colorscale='Turbo',
            connectgaps=True,
            contours=contoursd,
            colorbar={'title': dfcntr['Tool'].iloc[0]}
            ))
    #title = str(dfcntr['DateTime'].iloc[0]) + " Edge R²= " + str(r2)
    title = str(dfcntr['DateTime'].iloc[0])[:19]
    fig.update_layout(title={'text': title, 'font': {'size': 12}})
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=-150, y0=-150,
        x1=150, y1=150,
        opacity=0.2,
        fillcolor="blue",
        line_color="black",
        ) 
    return fig

if __name__ == '__main__':
    app.run_server(debug=True,port=8050)
