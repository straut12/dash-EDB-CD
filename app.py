import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures  # scikit-learn modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.stats.multicomp as multi

import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO
from dash import dcc, html, State
from dash import dash_table as dt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash.dependencies import Input, Output


# Dash apps are Flask apps

# https://dash.plotly.com/tutorial
# https://bootswatch.com/
# https://hellodash.pythonanywhere.com/
# https://hellodash.pythonanywhere.com/adding-themes/datatable
# https://community.plotly.com/t/styling-dash-datatable-select-rows-radio-button-and-checkbox/59466/3

#============IMPORT AND FORMAT DATA================

# Tried pd.to_datetime(df['_time'], format="%Y-%m-%d").dt.floor("d") but it left .0000000 for the H:M. May have been ok.
# DatetimeProperties.to_pydatetime is deprecated, in future version will return a Series containing python datetime objects instead of an ndarray. To retain the old behavior, call `np.array` on the result
df = pd.read_csv('CD-ALL-synthetic-data').assign(date=lambda data: pd.to_datetime(data["Date"], format="%Y-%m-%d")) # was not uploading with csv extension
# Updated to use google sheets csv

# Clean up. Get Date Time columns into correct format. Slot, Wfr, Lot as strings. And make sure feature/labels are at end of the table
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['DateTime'], format='%Y-%m-%d %H:%M:%S')
df['DateTime'] = df['DateTime'].dt.tz_localize('UTC')

df['Wfr'] = df['Wfr'].apply(lambda x: f'0{x}' if x < 10 else str(x))
df['Slot'] = df['Slot'].apply(lambda x: f'0{x}' if x < 10 else str(x))

for name, values in df.iloc[:, 2:14].items(): # Make sure category columns are strings
    df[name] = df[name].astype(str)

for name, values in df.iloc[:, 20:21].items(): # set Xmm/Ymm to 2 decimal places
    df[name] = df[name].round(2)

# Contour map hard coded for MP1. Would need extra checks to add MP2 and make sure modell columns are correct
modell = ['Xmm', 'Ymm', 'Rmm', 'MP1']  # make sure features/labels are at the end of the table for contour map modeling later on
for label in modell:
  col = df.pop(label)
  df[label] = col

target = df['Target'].iloc[0] # get initial target for default wafer maps
dflt_specl = [df['LS'].iloc[0], df['US'].iloc[0]] # get initial spec limits for default wafer maps
lotdflt1 = '23127034' # df['Lot'].iloc[0] # get initial lot,wfr for default contour wafer map1
wfrdflt1 = '16' # df['Wfr'].iloc[0]
lotdflt2 = '23128124' # df['Lot'].iloc[0] # get initial lot,wfr for default contour wafer map2
wfrdflt2 = '24' # df['Wfr'].iloc[0]
tooll = np.sort(df['Tool'].unique()).tolist() # get a unique list of tools for the graph selection menu and colors
lotl = np.sort(df['Lot'].unique()).tolist()
wfrl = np.sort(df['Wfr'].unique()).tolist()
# DO NOT USE 1,2,3 for labels. Can have confusing results due to int vs str scenarios
# +00:00 is Hour Min offset from UTC
# the freq/period parameter in pandas.date_range refers to the interval between dates generated. The default is "D", but it could be hourly, monthly, or yearly. 
#df['_time'] = pd.to_datetime(df['date'], unit='d', origin='1899-12-30') # Changed the decimal by a little and removed the +00:00
df['DateTime'] = pd.to_datetime((df['DateTime'])) # converted _time from obj to datetime64 with tz=UTC

dfcntr_dt = df.groupby(['DateTime', 'Tool', 'Lot', 'Wfr', 'CHUCK', 'PEB', 'DVLP'])['MP1'].agg(['mean', 'std']).reset_index() # selectable df table for contour plotsdfcntr_dt['DateTime'] = dfcntr_dt['DateTime'].dt.tz_convert(None)
dfcntr_dt.columns = ['DateTime', 'Tool', 'Lot', 'Wfr', 'CHUCK', 'PEB', 'DVLP', 'MP1A', 'MP1S']
dfcntr_dt['DateTime'] = dfcntr_dt['DateTime'].dt.tz_convert(None)
dfcntr_dt['MP1A'] = dfcntr_dt['MP1A'].round(1)
dfcntr_dt['MP1S'] = dfcntr_dt['MP1S'].round(1)

dfmask = df.groupby(['Tool', 'DEV', 'RETICLE', 'Tool-Mask']).agg({'DENSITY':'mean', 'FOCUSOFFSET':'mean', 'MASKCD':'mean', 'DOSE':'mean', 'MP1':'mean'}).reset_index()
# Calculate Tool-Mask offset compared to target
dfmask['MP1'] = target - dfmask['MP1']
dfmask['MP1'] = dfmask['MP1'].round(1)
# Compute the absolute values of 'MP1'
dfmask['MP1_abs'] = dfmask['MP1'].abs().round(1)
# Get the top 10 rows with the largest absolute values of 'MP1'
dfmasktop10 = dfmask.nlargest(10, 'MP1_abs')[['Tool', 'Tool-Mask', 'MP1', 'MP1_abs']].copy()
# Sort df_top10 by 'MP1_abs' in descending order
dfmasktop10 = dfmasktop10.sort_values('MP1_abs', ascending=False)

# Define a color map
color_map = {'CD101': '#636efa', 'CD102': '#ef553b', 'CD103': '#00cc96', 'CD104': '#ab63fa'}  # Replace with your actual tools and desired colors

# Map the 'Tool' column to colors
dfmasktop10['color'] = dfmasktop10['Tool'].map(color_map)

#============END IMPORT DATA================


max_table_rows = 11

#===START DASH AND CREATE LAYOUT OF TABLES/GRAPHS===============
# Create a bright and dark theme for the dash app. The theme is used for the tables and page background

# Create a custom dark theme for the charts. The color will match the dark theme color below
custom_template = pio.templates['plotly_dark'] 

theme_bright = dbc.themes.SANDSTONE
theme_chart_bright = pio.templates['seaborn']  # available plotly themes: simple_white, plotly, plotly_dark, ggplot2, seaborn, plotly_white, none

# available dash bootstrap dbc themes: BOOTSTRAP, CERULEAN, COSMO, CYBORG, DARKLY, FLATLY, JOURNAL, LITERA, LUMEN, LUX, MATERIA, MINTY, MORPH, PULSE, QUARTZ, SANDSTONE, SIMPLEX, SKETCHY, SLATE, SOLAR, SPACELAB, SUPERHERO, UNITED, VAPOR, YETI, ZEPHYR
darktheme = "SUPERHERO"
if darktheme == "SUPERHERO":
    theme_dark = dbc.themes.SUPERHERO
    custom_template.layout.paper_bgcolor = '#0f2537' # match the SUPERHERO theme
    custom_template.layout.plot_bgcolor = '#ced4da'  # light gray plot background color
elif darktheme == "SOLAR":
    theme_dark = dbc.themes.SOLAR
    custom_template.layout.paper_bgcolor = '#002b36' # match the SOLAR theme
    custom_template.layout.plot_bgcolor = '#ced4da'  # light gray plot background color
elif darktheme == "SLATE":
    theme_dark = dbc.themes.SLATE
    custom_template.layout.paper_bgcolor = '#272b30' # match the SOLAR theme
    custom_template.layout.plot_bgcolor = '#ced4da'  # light gray plot background color
elif darktheme == "DARKLY":
    theme_dark = dbc.themes.DARKLY
    custom_template.layout.paper_bgcolor = '#222222' # match the SOLAR theme
    custom_template.layout.plot_bgcolor = '#ced4da'  # light gray plot background color

pio.templates['custom_dark'] = custom_template
theme_chart_dark = pio.templates['custom_dark']

tukey_table_cell_highlight = '#cfb974'  # Color for highlighting a tukeyHSD flagged cell

# background color -> #002b36    # gray dark
# 'backgroundColor': 'rgb(40, 40, 40)' # Table header color
# 'backgroundColor': '#052027',         # Color for highlighting a tukeyHSD flagged cell
# radio button color -> #b58900  # dark gold
# slider bar color  -> #cfb974   # light gold

#===START DASH AND CREATE LAYOUT OF TABLES/GRAPHS================

app = dash.Dash(__name__, external_stylesheets=[theme_dark])

title = html.H1("PROCESS DASHBOARD", style={'font-size': '18px'}) #'color': 'white',

theme_switch = ThemeSwitchAIO(
    aio_id="theme", themes=[theme_dark, theme_bright]
)

calendar = html.Div(["Date Range ",
        dcc.DatePickerRange(
            id="date-range",
            min_date_allowed=df["date"].min().date(),
            max_date_allowed=df["date"].max().date(),
            start_date=df["date"].min().date(),
            end_date=df["date"].max().date(),
        )])

tukey_radio = html.Div(["Tukey HSD ",
        dcc.RadioItems(
        id='tukey-radio', 
        options=[{'value': x, 'label': x}  # radio button labels and values
                 for x in ['ON', 'OFF']],  # radio button labels and values
        value='OFF',   # Default
        labelStyle={'display': 'inline-block'}
        )])

summary_tableh = html.Div(["Summary Table", dt.DataTable(id='summary-table',
        columns=[
            {'name': ['Summary Table', 'No'], 'id': 'no_rows'},
            {'name': ['Summary Table','Tool'], 'id': 'tool'},
            {'name': ['Summary Table','Count'], 'id': 'count'},
            {'name': ['Summary Table','Mean'], 'id': 'mean'},
            {'name': ['Summary Table','Sigma'], 'id': 'sigma'}
        ],
        data=[{'no_rows': i} for i in range(1,max_table_rows)],
        sort_action='native',
        sort_mode='multi',
        editable=False,
        merge_duplicate_headers=True,
        style_cell={'textAlign': 'center'},
        style_header={          #'backgroundColor': 'rgb(40, 40, 40)',
            'fontWeight': 'bold'
        })
        ])  # style={'display': 'inline-table', 'margin':'10px', 'width': '20%'}

tukey_tableh = html.Div(["Top 10 Tukey HSD Results (each table update may take 3-5sec)", dt.DataTable(id='tukey-table',
        columns=[
            {'name': ['Test Result on Means', 'No'], 'id': 'no_rows'},
            {'name': ['Test Result on Means','Tool'], 'id': 'tukey-ave-tool'},
            {'name': ['Test Result on Means','Grp1'], 'id': 'tukey-ave-grp1'},
            {'name': ['Test Result on Means','Grp2'], 'id': 'tukey-ave-grp2'},
            {'name': ['Test Result on Means','AvgDlt'], 'id': 'tukey-ave-meandiff'},
            {'name': ['Test Result on Means','p-Adj'], 'id': 'tukey-ave-padj'},
            {'name': ['Test Result on Sigmas','No'], 'id': 'no_rows'},
            {'name': ['Test Result on Sigmas','Tool'], 'id': 'tukey-sig-tool'},
            {'name': ['Test Result on Sigmas','Grp1'], 'id': 'tukey-sig-grp1'},
            {'name': ['Test Result on Sigmas','Grp2'], 'id': 'tukey-sig-grp2'},
            {'name': ['Test Result on Sigmas','AvgDlt'], 'id': 'tukey-sig-meandiff'},
            {'name': ['Test Result on Sigmas','p-Adj'], 'id': 'tukey-sig-padj'}
        ],
        data=[{'no_rows': i} for i in range(1,max_table_rows)],
        style_data_conditional=[
            {
                'if': {
                    'column_id': 'tukey-ave-grp1',
                    'filter_query': '{tukey-ave-grp1} contains "**"'
                },
                'backgroundColor': tukey_table_cell_highlight,
            },
            {
                'if': {
                    'column_id': 'tukey-ave-grp2',
                    'filter_query': '{tukey-ave-grp2} contains "**"'
                },
                'backgroundColor': tukey_table_cell_highlight,
            },
            {
                'if': {
                    'column_id': 'tukey-sig-grp1',
                    'filter_query': '{tukey-sig-grp1} contains "**"'
                },
                'backgroundColor': tukey_table_cell_highlight,
            },
            {
                'if': {
                    'column_id': 'tukey-sig-grp2',
                    'filter_query': '{tukey-sig-grp2} contains "**"'
                },
                'backgroundColor': tukey_table_cell_highlight,
            }
        ],
        editable=False,
        sort_action='native',
        sort_mode='multi',
        style_cell={'textAlign': 'center'},
        merge_duplicate_headers=True,
        style_header={          #'backgroundColor': 'rgb(40, 40, 40)',
            'fontWeight': 'bold',
        })
        ])

chart_range_slider = html.Div([dcc.RangeSlider(55, 75, 0.5, value=dflt_specl, tooltip={"placement": "bottom", "always_visible": False}, id='limit-slider')])

chart_mpx_radio = html.Div(
    dcc.RadioItems(
        id='chart-y', 
        options=[{'value': x, 'label': x}  # radio button labels and values
                 for x in ['MP1', 'MP2']],  # radio button labels and values
        value='MP1',   # Default
        labelStyle={'display': 'inline-block'}
        ))

boxplt_mpx_radio = html.Div(
        dcc.RadioItems(
        id='boxplt-y', 
        options=[{'value': x, 'label': x}  # radio button labels and values
                 for x in ['MP1', 'MP2']],  # radio button labels and values
        value='MP1',   # Default
        labelStyle={'display': 'inline-block'}
        ))

tool_checklist = html.Div(dcc.Checklist(
        id="tool_list",  # id names will be used by the callback to identify the components
        options=tooll, # list of the tools
        value=tooll, # default selections
        inline=True))

unit_list_radio = html.Div(
        dcc.RadioItems(
        id='unit', 
        options=[{'value': x, 'label': x}  # radio button labels and values
                for x in ['ARC', 'COAT', 'CHUCK', 'PEB', 'DVLP']],  # radio button labels and values
        value='PEB',   # Default
        labelStyle={'display': 'inline-block'}
        ))

line_chart1 = html.Div([dcc.Graph(figure={}, id='line-chart1')])  # figure is blank dict because created in callback below

boxplot1 = html.Div([dcc.Graph(figure={}, id='box-plot1')])

cntr1_radio = html.Div(
    dcc.RadioItems(
        id='cntr1-radio', 
        options=[{'value': x, 'label': x}  # radio button labels and values
                 for x in ['Auto', 'Manual']],  # radio button labels and values
        value='Auto',   # Default
        labelStyle={'display': 'inline-block'}
        ))

cntr2_radio = html.Div(
    dcc.RadioItems(
        id='cntr2-radio', 
        options=[{'value': x, 'label': x}  # radio button labels and values
                 for x in ['Auto', 'Manual']],  # radio button labels and values
        value='Auto',   # Default
        labelStyle={'display': 'inline-block'}
        ))

cntr1_range_slider = html.Div([dcc.RangeSlider(60, 70, 0.5, value=[60, 70], tooltip={"placement": "bottom", "always_visible": False}, id='cntr1-slider')]),

cntr2_range_slider = html.Div([dcc.RangeSlider(60, 70, 0.5, value=[60, 70], tooltip={"placement": "bottom", "always_visible": False}, id='cntr2-slider')]),

lot1_dd = html.Div([dcc.Dropdown(lotl, lotdflt1, id='lot1-dd')]),

wfr1_dd = html.Div([dcc.Dropdown(wfrl, wfrdflt1, id='wfr1-dd')]),

lot2_dd = html.Div([dcc.Dropdown(lotl, lotdflt2, id='lot2-dd')]),

wfr2_dd = html.Div([dcc.Dropdown(wfrl, wfrdflt2, id='wfr2-dd')]),

cntr_dt_deselect_btn = html.Div([html.Button('Deselect', id='cntr-dt-deselect-btn')]),

cntr_plot1 = html.Div([dcc.Graph(figure={}, id='cntr1')]),

cntr_plot2 = html.Div([dcc.Graph(figure={}, id='cntr2')]), # , style={'display': 'inline-block', 'width': 430}

contour_table = html.Div([
    dt.DataTable(
        id='cntr-table',
        columns=[{'name': col, 'id': col} for col in dfcntr_dt.columns],
        data=dfcntr_dt.to_dict('records'),
        #page_size=5,
        style_table={'height': '380px', 'overflowX': 'auto', 'overflowY': 'auto'},
        row_selectable='single',
        fixed_rows={'headers': True, 'data': 0},
        sort_action='native',
        sort_mode='multi',
        style_cell={'textAlign': 'center'}
    )])

reticle_mpx_radio = html.Div(["Reticle Analysis  ",
    dcc.RadioItems(
    id='reticle-y', 
    options=[{'value': x, 'label': x}  # radio button labels and values
                for x in ['MP1', 'MP2']],  # radio button labels and values
    value='MP1',   # Default
    labelStyle={'display': 'inline-block'}
    )]),

dose2size_maskCD_chart = html.Div([dcc.Graph(figure={}, id='dose2size_maskCD')]),

focus_offset_mask_density_chart = html.Div([dcc.Graph(figure={}, id='focus_offset_mask_density')]),

reticle_offset_pareto = html.Div([dcc.Graph(figure={}, id='reticle_offset')]),

reticle_analysis_chart = html.Div([dcc.Graph(figure={}, id='reticle')]),

heatmap_chart = html.Div([dcc.Graph(figure={}, id='sigma-heat-map')])

footer = html.P(
                [
                    html.Span('Created by Sean Trautman  ', className='mr-2'),
                    html.A(html.I(className='fas fa-envelope-square mr-1'), href='mailto:s.trautman12@gmail.com'),
                    html.A(html.I(className='fab fa-linkedin mr-1'), href='https://www.linkedin.com/in/sean-trautman-b66bb8175/')
                ], 
                className='lead')

# Layout of the dash graphs, tables, drop down menus, etc
# Using dbc container for styling/formatting
app.layout = dbc.Container([
    # Summary and Tukey Tables
    dbc.Row([
        dbc.Col(title, width={"size":9, "justify":"between"}),
        dbc.Col(theme_switch, width={"size":3, "justify":"between"})]),
    dbc.Row([
        dbc.Col(calendar, width={"size":5, "justify":"left"}),
        dbc.Col(tukey_radio, width={"size":7, "justify":"between"})]),
    dbc.Row([
        dbc.Col(summary_tableh, width={"size":4}),
        dbc.Col(tukey_tableh, width={"size":8})]),
    # Charts and Boxplot
    dbc.Row([
        dbc.Col(chart_range_slider, width={"size":12})]),
    dbc.Row([
        dbc.Col(chart_mpx_radio, width={"size":6}),
        dbc.Col(boxplt_mpx_radio, width={"size":6})]),
    dbc.Row([
        dbc.Col(tool_checklist, width={"size":6}),
        dbc.Col(unit_list_radio, width={"size":6})]),
    dbc.Row([
        dbc.Col(line_chart1, width={"size":6}),
        dbc.Col(boxplot1, width={"size":6})]),
    # Contour Plots
    dbc.Row([
        dbc.Col(cntr1_radio, width={"size":3}),
        dbc.Col(cntr2_radio, width={"size":3})]),
    dbc.Row([
        dbc.Col(cntr1_range_slider, width={"size":3}),
        dbc.Col(cntr2_range_slider, width={"size":3})]),
    dbc.Row([
        dbc.Col(lot1_dd, width={"size":2}),
        dbc.Col(wfr1_dd, width={"size":1}),
        dbc.Col(lot2_dd, width={"size":2}),
        dbc.Col(wfr2_dd, width={"size":1}),
        dbc.Col(cntr_dt_deselect_btn, width={"size":1})]),
    dbc.Row([
        dbc.Col(cntr_plot1, width=3, style={'width': '430px'}),
        dbc.Col(cntr_plot2, width=3, style={'width': '430px'}),
        dbc.Col(contour_table, width={"size":6})]),
    # Reticle and Focus Analysis
    dbc.Row([
        dbc.Col(dose2size_maskCD_chart, width={"size":3}),
        dbc.Col(focus_offset_mask_density_chart, width={"size":3}),
        dbc.Col(reticle_offset_pareto, width={"size":3})]),
    # Reticle boxplot and Heatmap
    dbc.Row([
        dbc.Col(reticle_mpx_radio, width={"size":12})]),
    dbc.Row([
        dbc.Col(reticle_analysis_chart, width={"size":6}),
        dbc.Col(heatmap_chart, width={"size":6})]),
    dbc.Row([
        dbc.Col(footer, width={"size":12})])
    ], fluid=True, className="dbc dbc-row-selectable")


#=====CREATE INTERACTIVE GRAPHS=============
# Callbacks are used to update the graphs and tables when the user changes the inputs   
# Box plot for reticle analysis
@app.callback(
    Output("reticle", "figure"), 
    Input("reticle-y", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def generate_bx_reticle(y, start_date, end_date, toggle):
    chart_theme = theme_chart_dark if toggle else theme_chart_bright
    filtered_data = df.query("date >= @start_date and date <= @end_date")
    fig = px.box(filtered_data, x="Tool", y=y, color="RETICLE", notched=True, template=chart_theme, hover_data=[filtered_data['Lot'], filtered_data['Wfr'],  filtered_data['DEV']], category_orders={"Tool": tooll})
    fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Pareto for Reticle Offset
@app.callback(
    Output("reticle_offset", "figure"), 
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def generate_reticle_offset_pareto(toggle):
    chart_theme = theme_chart_dark if toggle else theme_chart_bright
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dfmasktop10['Tool-Mask'], y=dfmasktop10['MP1'], name='Reticle Offset', marker_color=dfmasktop10['color']))
    # Set up the layout with y-axes
    fig.update_layout(
        yaxis=dict(title='CD Delta to Target (nm)'),
        template=chart_theme,
        title='Top 10 Reticles with Offset from Target',
        #yaxis2=dict(title='Cumulative Percentage', overlaying='y', side='right')
    )
    return fig

# Scatter plot for Mask CD
@app.callback(
    Output("dose2size_maskCD", "figure"), 
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def generate_mask_cd_chart(toggle):
    chart_theme = theme_chart_dark if toggle else theme_chart_bright
    fig = px.scatter(dfmask, x='MASKCD', y='DOSE', color='Tool', template=chart_theme,
                     labels={
                     "MASKCD": "Mask CD (nm)",
                     "DOSE": "Dose-to-Target (mJ)"
                 })
    fig.update_layout(title_text='Reticle Dose vs Mask CD')
    return fig

# Scatter plot for Focus offset vs pattern density
@app.callback(
    Output("focus_offset_mask_density", "figure"), 
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def generate_mask_cd_chart(toggle):
    chart_theme = theme_chart_dark if toggle else theme_chart_bright
    fig = px.scatter(dfmask, x='DENSITY', y='FOCUSOFFSET', color='Tool', template=chart_theme,
                     labels={
                     "DENSITY": "Through-out Rate %",
                     "FOCUSOFFSET": "Focus Offset (um)"
                 })
    fig.update_layout(title_text='Reticle Focus Offsets')
    return fig

# Heat map for CD sigma analysis
@app.callback(
    Output("sigma-heat-map", "figure"),
    Input("tool_list", "value"), 
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("chart-y", "value"),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def generate_hm_cd_sigma(tool, start_date, end_date, mpx, toggle):
    # Create heat map for CD sigma analysis. A dictionary of df's is created for each unit and then concatenated into one df
    # Heat map is created by grouping by date and Tool_unit and then taking the std of MP1
    chart_theme = theme_chart_dark if toggle else theme_chart_bright
    unitl = ['ARC', 'COAT', 'CHUCK', 'PEB', 'DVLP']
    heatmapdfd = {}
    for unit in unitl:
        heatmapdfd[unit] = df[['DateTime', 'date', 'Tool', unit, mpx]]
        heatmapdfd[unit] = heatmapdfd[unit].rename(columns={unit: 'unit'})
    heatmapdf = pd.concat(heatmapdfd.values(), ignore_index=True)
    heatmapdf['Tool_unit'] = heatmapdf['Tool'] + '_' + heatmapdf['unit']
    heatmapdf = heatmapdf.groupby(['date', 'Tool_unit'])[mpx].apply(np.std).reset_index()
    heatmap = go.Heatmap(z=heatmapdf[mpx], x=heatmapdf['date'], y=heatmapdf['Tool_unit'], colorscale='RdBu_r')
    # create the layout with categoryorder parameter
    layout = go.Layout(title='CD Sigma Heatmap', template=chart_theme, xaxis=dict(categoryorder='category ascending'), yaxis=dict(categoryorder='category ascending'))
    # create the figure
    fig = go.Figure(data=[heatmap], layout=layout)
    return fig

# Summary table update 
@app.callback(
    Output('summary-table', 'data'),     # args are component id and then component property. component property is passed
    Input('date-range', 'start_date'),  # in order to the chart function below
    Input('date-range', 'end_date'),
    State('summary-table', 'data'),
    Input('chart-y', 'value'))
def summary_table(start_date, end_date, rows, mpx):
    filtered_data = df.query("date >= @start_date and date <= @end_date")
    dfsummary = filtered_data.groupby('Tool')[mpx].describe()  
    dfsummary = dfsummary.reset_index()
    dfsummary = dfsummary.drop(['min', '25%', '50%', '75%', 'max'], axis=1)
    dfsummary.loc[:, "mean"] = dfsummary["mean"].map('{:.1f}'.format)
    dfsummary.loc[:, "std"] = dfsummary["std"].map('{:.1f}'.format)
    summaryd = {'tool':'Tool', 'count':'count', 'mean':'mean', 'sigma':'std'}
    for i, row in enumerate(rows):
        for key, value in summaryd.items():
            try:
                row[key] = dfsummary.at[i, value]
            except:
                row[key] = ''
    return rows

# Tukey HSD table update for mean and sigma difference
@app.callback(
    Output('tukey-table', 'data'),
    Input("tool_list", "value"),        # args are component id and then component property. component property is passed
    Input("date-range", "start_date"),  # in order to the chart function below
    Input("date-range", "end_date"),
    Input("tukey-radio", "value"),
    State('tukey-table', 'data'),
    Input("chart-y", "value"))
def tukey_table(tool, start_date, end_date, onoff, rows, mpx):
    # Tables for Tukey HSD results
    tukey_aved = {'tukey-ave-tool':'Tool', 'tukey-ave-grp1':'group1', 'tukey-ave-grp2':'group2', 'tukey-ave-meandiff':'meandiff','tukey-ave-padj':'p-adj'}
    tukey_sigd = {'tukey-sig-tool':'Tool', 'tukey-sig-grp1':'group1', 'tukey-sig-grp2':'group2', 'tukey-sig-meandiff':'meandiff', 'tukey-sig-padj':'p-adj'}    
    if onoff == 'OFF':                               # Initialize table with empty values when turned OFF
        for i, row in enumerate(rows):
            for key, value in tukey_aved.items():
                row[key] = ''
            for key, value in tukey_sigd.items():
                row[key] = ''
        return rows
    else:
        filtered_data = df.query("date >= @start_date and date <= @end_date")  # If table turned ON then update with values
        mask = filtered_data.Tool.isin(tool)
        dftooll = filtered_data[mask]

        # get pairwise tukey HSD results for mean difference
        tukey_results = pd.DataFrame(columns=['Tool'])
        unitl = ['ARC', 'COAT', 'CHUCK', 'PEB', 'DVLP']
        for tool in dftooll['Tool'].unique():
            for unit in unitl:
                df_tool = dftooll[dftooll['Tool'] == tool]
                tukey = multi.pairwise_tukeyhsd(df_tool[mpx], df_tool[unit])
                tukey_df = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                tukey_df['Tool'] = tool
                tukey_results = pd.concat([tukey_results, tukey_df], ignore_index=True)
        tukey_results = tukey_results[tukey_results['reject'] != False]  # remove rows where reject is False since not significant
        dftukey_ave = tukey_results.sort_values(['Tool', 'group1']).drop(['lower', 'upper', 'reject'], axis=1).reset_index()
        dftukey_ave = dftukey_ave.round(2)
        tukeytbld = {}
        grpl = ['group1', 'group2']
        for grp in grpl:
            tukeytbld[grp] = dftukey_ave[['Tool', grp]]
            tukeytbld[grp] = tukeytbld[grp].rename(columns={grp: 'group'})
        dftukeyhl = pd.concat(tukeytbld.values(), ignore_index=True)
        dftukeyhl['Tool-group'] = dftukeyhl['Tool'] + '_' + dftukeyhl['group']
        group_counts = dftukeyhl['Tool-group'].value_counts()
        group_ave = group_counts[group_counts > 1].index.tolist()
        # look for matches and update the values
        for i in range(len(dftukey_ave)):
            if dftukey_ave['Tool'][i] + '_' + dftukey_ave['group1'][i] in group_ave:
                dftukey_ave.at[i, 'group1'] += '**'
        for i in range(len(dftukey_ave)):
            if dftukey_ave['Tool'][i] + '_' + dftukey_ave['group2'][i] in group_ave:
                dftukey_ave.at[i, 'group2'] += '**'

        # get pairwise tukey HSD results for sigma difference
        dftooll['Lot_Wfr'] = dftooll.apply(lambda row: f"{row['Lot']}-{row['Wfr']}", axis=1) 
        grouped = dftooll.groupby(['Lot_Wfr', 'Tool', 'DVLP', 'PEB', 'CHUCK', 'ARC', 'COAT'])
        dfsigmas = grouped[mpx].apply(lambda x: np.std(x))  # calculate the sigma of each wfr
        dfsigmas = dfsigmas.reset_index()
        tukey_results = pd.DataFrame(columns=['Tool'])
        for tool in dfsigmas['Tool'].unique():
            for unit in unitl:
                df_tool = dfsigmas[dfsigmas['Tool'] == tool]
                tukey = multi.pairwise_tukeyhsd(df_tool[mpx], df_tool[unit])
                tukey_df = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                tukey_df['Tool'] = tool
                tukey_results = pd.concat([tukey_results, tukey_df], ignore_index=True)
        tukey_results = tukey_results[tukey_results['reject'] != False]  # remove rows where reject is False since not significant
        dftukey_sig = tukey_results.sort_values(['Tool', 'group1']).drop(['lower', 'upper', 'reject'], axis=1).reset_index()
        dftukey_sig = dftukey_sig.round(2)
        tukeytbld = {}
        for grp in grpl:
            tukeytbld[grp] = dftukey_sig[['Tool', grp]]
            tukeytbld[grp] = tukeytbld[grp].rename(columns={grp: 'group'})
        dftukeyhl = pd.concat(tukeytbld.values(), ignore_index=True)
        dftukeyhl['Tool-group'] = dftukeyhl['Tool'] + '_' + dftukeyhl['group']
        group_counts = dftukeyhl['Tool-group'].value_counts()
        group_sig = group_counts[group_counts > 1].index.tolist()
        for i in range(len(dftukey_sig)):
            if dftukey_sig['Tool'][i] + '_' + dftukey_sig['group1'][i] in group_sig:
                dftukey_sig.at[i, 'group1'] += '**'
        for i in range(len(dftukey_sig)):
            if dftukey_sig['Tool'][i] + '_' + dftukey_sig['group2'][i] in group_sig:
                dftukey_sig.at[i, 'group2'] += '**'

        for i, row in enumerate(rows):
            for key, value in tukey_aved.items():
                try:
                    row[key] = dftukey_ave.at[i, value]
                except:
                    row[key] = ''
            for key, value in tukey_sigd.items():
                try:
                    row[key] = dftukey_sig.at[i, value]
                except:
                    row[key] = ''
        return rows

# Create plotly express line chart
@app.callback(
    Output("line-chart1", "figure"),    # args are component id and then component property
    Input("tool_list", "value"),        # args are component id and then component property. component property is passed
    Input("date-range", "start_date"),  # in order to the chart function below
    Input("date-range", "end_date"),
    Input("chart-y", "value"),
    Input("limit-slider", "value"),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def update_line_chart(tool, start_date, end_date, mpx, limits, toggle):    # callback function arg 'tool' refers to the component property of the input or "value" above
    chart_theme = theme_chart_dark if toggle else theme_chart_bright
    filtered_data = df.query("date >= @start_date and date <= @end_date")  # Get only data within time frame selected
    mask = filtered_data.Tool.isin(tool)                                   # Create a panda series with True/False of only tools selected 
    fig = px.line(filtered_data[mask],   
        x='DateTime', y=mpx, color='Tool'
        ,category_orders={'Tool':tooll}  # can manually set colors color_discrete_sequence = ['darkred', 'dodgerblue', 'green', 'tan']
        ,color_discrete_sequence = ['darkorange', 'dodgerblue', 'green', 'darkviolet']
        ,line_shape="hv"
        ,hover_data=['Lot', 'Wfr', 'DEV', 'RETICLE', 'Slot', 'ARC', 'COAT', 'CHUCK', 'PEB', 'DVLP', 'Site', 'Xmm', 'Ymm']
        ,markers=True,
        template=chart_theme)
    fig.update_traces(mode="markers")
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Create plotly express box plot
@app.callback(
    Output("box-plot1", "figure"), 
    Input("boxplt-y", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("unit", "value"),
    Input("limit-slider", "value"),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def generate_bx_unit(y, start_date, end_date, unit, limits, toggle):
    chart_theme = theme_chart_dark if toggle else theme_chart_bright
    filtered_data = df.query("date >= @start_date and date <= @end_date")
    fig = px.box(filtered_data, x="Tool", y=y, color=unit, notched=True, template=chart_theme, hover_data=[filtered_data['Lot'], filtered_data['Wfr'],  filtered_data['Site']], category_orders={"Tool": tooll})
    fig.add_hline(y=limits[0], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=limits[1], line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=target, line_width=1, line_dash="dash", line_color="black")
    return fig

# Create plotly go (graphical objects) for contour plots
# Contour plot 1
@app.callback(
    Output("cntr1", "figure"), 
    Input("lot1-dd", "value"),
    Input("wfr1-dd", "value"),
    Input("cntr1-radio", "value"),
    Input("cntr1-slider", "value"),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"))
def generate_cntr_1(lotID, wfrID, radio, cntr_limits, toggle):
    chart_theme = theme_chart_dark if toggle else theme_chart_bright
    dfcntr = df.loc[(df['Lot'] == lotID) & (df['Wfr'] == wfrID )]
    dfcntr = dfcntr.drop(['Date', 'date', 'Target','LS','US'], axis=1)
    # Create model to predict MP1 where there was no measurement
    features = dfcntr.iloc[:, -4:-2].values
    label = dfcntr.iloc[:, -1].values
    features_train, features_test, label_train, label_test = train_test_split(features, label, test_size = 0.2, random_state=0)
    min_poly_degree = 2 # loop through poly degrees to find the best fit
    max_poly_degree = 6
    min_rmse = float('inf')
    for poly_degree in range (min_poly_degree, max_poly_degree+1):
        poly_reg = PolynomialFeatures(degree = poly_degree)
        X_poly = poly_reg.fit_transform(features_train)
        regressor = LinearRegression()
        regressor.fit(X_poly, label_train)
        # Predicting the Test set results
        label_pred = regressor.predict(poly_reg.transform(features_test))
        #print(np.concatenate((label_pred.reshape(len(label_pred),1), label_test.reshape(len(label_test),1)),1))
        # Evaluating Model Performance
        rmse = np.sqrt(mean_squared_error(label_test, label_pred))
        if rmse < min_rmse:
            min_pd = poly_degree
            min_rmse = rmse        
    poly_reg = PolynomialFeatures(degree = min_pd) # use the best fitting poly degree for final model
    X_poly = poly_reg.fit_transform(features_train)
    regressor = LinearRegression()
    regressor.fit(X_poly, label_train)
    # Predicting the Test set results
    label_pred = regressor.predict(poly_reg.transform(features_test))
    # Create 2D matrix: X, Y, Z for contour plot
    xcoord = np.append(dfcntr.iloc[:, -4:-3].values, [i for i in range(-150, 160, 10)]) # create dummy X/Y on the edge and append to the xmm/ymm lists for better edge coverage of predictions
    ycoord = np.append(dfcntr.iloc[:, -3:-2].values, [i for i in range(-150, 160, 10)])
    xmm = np.sort(np.unique(xcoord))
    ymm = np.sort(np.unique(ycoord))
    #xmm = np.sort(np.unique(dfcntr.iloc[:, -4:-3].values)) # get the unique Xmm location values
    #ymm = np.sort(np.unique(dfcntr.iloc[:, -3:-2].values)) # get the unique Ymm location values
    X,Y = np.meshgrid(xmm, ymm)
    Z = np.zeros((X.shape))
    
    # Create a dict that maps all the measured MP1s with their X,Y loc
    dict = dfcntr.to_dict('list')
    MP1ave = dfcntr['MP1'].mean()  # MP1ave will be used to set the Z value outside the wafer diam for countour plotting purposes
    MP1_map={}
    for i in range(len(dict['MP1'])):
        MP1_map[str(dict['Xmm'][i]) + '-' + str(dict['Ymm'][i])] = dict['MP1'][i]
    # Create a full 2D map of MP1 for every X/Y loc. Use meas values if present, otherwise fill in with predicted values
    rmax = dfcntr['Rmm'].max()    # will only predict values beyond the radius of measured values. will let plotly contour fill in the middle of the wafer
    for i in range(Z.shape[1]):
        for j in range(Z.shape[0]):
            Xloc = X[0][i]
            Yloc = Y[j][0]
            if MP1_map.get(str(Xloc) + '-' + str(Yloc)) == None:
                radius = np.sqrt(Xloc**2 + Yloc**2)
                if radius > rmax and radius < 150:
                    pred_value = regressor.predict(poly_reg.transform([[Xloc, Yloc]]))
                    Z[j][i] = pred_value[0]
                    #print("no key {} radius: {:.0f} Z:{:.1f} ".format(str(Xloc) + '-' + str(Yloc), radius, Z[j][i]))
                elif radius > 150:
                    Z[j][i] = MP1ave
                    #print("pred value for {} is {} at radius {:.0f}".format(str(Xloc) + '-' + str(Yloc), Z[i][j], radius))
            else:
                    #print("key {} has th {}".format(str(Xloc) + '-' + str(Yloc), th_map[str(Xloc) + '-' + str(Yloc)]))
                    Z[j][i] = MP1_map[str(Xloc) + '-' + str(Yloc)]
    Z = np.where(Z==0, np.nan, Z) # replace 0's with nan. This is primarily in the wafer area where there wasn't a measurement
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
            colorbar={'title': dfcntr['MP'].iloc[0]})
            )
    title = str(dfcntr['DateTime'].iloc[0])[:19] + ' Lot: ' + dfcntr['Lot'].iloc[0] + ' Wfr: ' + dfcntr['Wfr'].iloc[0] + ' Slot:' + dfcntr['Slot'].iloc[0] + '<br>Tool: ' + dfcntr['Tool'].iloc[0] + ' ' + dfcntr['COAT'].iloc[0] + ' ' + dfcntr['CHUCK'].iloc[0] + ' ' + dfcntr['PEB'].iloc[0] + ' ' + dfcntr['DVLP'].iloc[0] + ' ARC: ' + dfcntr['ARC'].iloc[0] +'<br>Dev: ' + dfcntr['DEV'].iloc[0] + ' Mask: ' + dfcntr['RETICLE'].iloc[0]  + ' ERMSE: ' + str(round(min_rmse, 2)) + 'nm (' + str(min_pd) + ')'
    fig.update_layout(title={'text': title, 'font': {'size': 12}}, template=chart_theme)
    fig.update_xaxes(title_text='Xmm')
    fig.update_yaxes(title_text='Ymm')
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=-150, y0=-150,
        x1=150, y1=150,
        opacity=0.2,
        fillcolor="blue",
        line_color="black",
        ) 
    return fig

# Define a callback to deselect the row when the deselect button is clicked
@app.callback(
    Output('cntr-table', 'selected_rows'),
    Input('cntr-dt-deselect-btn', 'n_clicks')
)
def update_selected_rows(n_clicks):
    return []

# 2nd Contour Map
@app.callback(
    Output("cntr2", "figure"),
    Input("lot2-dd", "value"),
    Input("wfr2-dd", "value"),
    Input("cntr2-radio", "value"),
    Input("cntr2-slider", "value"),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"),
    Input("cntr-table", "selected_rows"),
    State("cntr-table", "data"))
def generate_cntr_2(lotID, wfrID, radio, cntr_limits, toggle, selected_rows, data):
    chart_theme = theme_chart_dark if toggle else theme_chart_bright
    if selected_rows:
        lotID = data[selected_rows[0]]['Lot']
        wfrID = data[selected_rows[0]]['Wfr']
    dfcntr = df.loc[(df['Lot'] == lotID) & (df['Wfr'] == wfrID )]
    dfcntr = dfcntr.drop(['Date', 'date','Target','LS','US'], axis=1)
    # Create model to predict MP1 where there was no measurement
    features = dfcntr.iloc[:, -4:-2].values
    label = dfcntr.iloc[:, -1].values
    features_train, features_test, label_train, label_test = train_test_split(features, label, test_size = 0.2, random_state=0)
    min_poly_degree = 2 # loop through poly degrees to find the best fit
    max_poly_degree = 6
    min_rmse = float('inf')
    for poly_degree in range (min_poly_degree, max_poly_degree+1):
        poly_reg = PolynomialFeatures(degree = poly_degree)
        X_poly = poly_reg.fit_transform(features_train)
        regressor = LinearRegression()
        regressor.fit(X_poly, label_train)
        # Predicting the Test set results
        label_pred = regressor.predict(poly_reg.transform(features_test))
        #print(np.concatenate((label_pred.reshape(len(label_pred),1), label_test.reshape(len(label_test),1)),1))
        # Evaluating Model Performance
        rmse = np.sqrt(mean_squared_error(label_test, label_pred))
        if rmse < min_rmse:
            min_pd = poly_degree
            min_rmse = rmse        
    poly_reg = PolynomialFeatures(degree = min_pd) # use the best fitting poly degree for final model
    X_poly = poly_reg.fit_transform(features_train)
    regressor = LinearRegression()
    regressor.fit(X_poly, label_train)
    # Predicting the Test set results
    label_pred = regressor.predict(poly_reg.transform(features_test))
    # Create 2D matrix: X, Y, Z for contour plot
    xcoord = np.append(dfcntr.iloc[:, -4:-3].values, [i for i in range(-150, 160, 10)]) # create dummy X/Y on the edge and append to the xmm/ymm lists for better edge coverage of predictions
    ycoord = np.append(dfcntr.iloc[:, -3:-2].values, [i for i in range(-150, 160, 10)])
    xmm = np.sort(np.unique(xcoord))
    ymm = np.sort(np.unique(ycoord))
    X,Y = np.meshgrid(xmm, ymm)
    Z = np.zeros((X.shape))
    
    # Create a dict that maps all the measured MP1s with their X,Y loc
    dict = dfcntr.to_dict('list')
    MP1ave = dfcntr['MP1'].mean()  # MP1ave will be used to set the Z value outside the wafer diam for countour plotting purposes
    MP1_map={}
    for i in range(len(dict['MP1'])):
        MP1_map[str(dict['Xmm'][i]) + '-' + str(dict['Ymm'][i])] = dict['MP1'][i]
    # Create a full 2D map of MP1 for every X/Y loc. Use meas values if present, otherwise fill in with predicted values
    rmax = dfcntr['Rmm'].max()    # will only predict values beyond the radius of measured values. will let plotly contour fill in the middle of the wafer
    for i in range(Z.shape[1]):
        for j in range(Z.shape[0]):
            Xloc = X[0][i]
            Yloc = Y[j][0]
            if MP1_map.get(str(Xloc) + '-' + str(Yloc)) == None:
                radius = np.sqrt(Xloc**2 + Yloc**2)
                if radius > rmax and radius < 150:
                    pred_value = regressor.predict(poly_reg.transform([[Xloc, Yloc]]))
                    Z[j][i] = pred_value[0]
                    #print("no key {} radius: {:.0f} Z:{:.1f} ".format(str(Xloc) + '-' + str(Yloc), radius, Z[j][i]))
                elif radius > 150:
                    Z[j][i] = MP1ave
                    #print("pred value for {} is {} at radius {:.0f}".format(str(Xloc) + '-' + str(Yloc), Z[i][j], radius))
            else:
                    #print("key {} has th {}".format(str(Xloc) + '-' + str(Yloc), th_map[str(Xloc) + '-' + str(Yloc)]))
                    Z[j][i] = MP1_map[str(Xloc) + '-' + str(Yloc)]
    Z = np.where(Z==0, np.nan, Z) # replace 0's with nan. This is primarily in the wafer area where there wasn't a measurement
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
            colorbar={'title': dfcntr['MP'].iloc[0]}
            ))
    title = str(dfcntr['DateTime'].iloc[0])[:19] + ' Lot: ' + dfcntr['Lot'].iloc[0] + ' Wfr: ' + dfcntr['Wfr'].iloc[0] + ' Slot:' + dfcntr['Slot'].iloc[0] + '<br>Tool: ' + dfcntr['Tool'].iloc[0] + ' ' + dfcntr['COAT'].iloc[0] + ' ' + dfcntr['CHUCK'].iloc[0] + ' ' + dfcntr['PEB'].iloc[0] + ' ' + dfcntr['DVLP'].iloc[0] + ' ARC: ' + dfcntr['ARC'].iloc[0] +'<br>Dev: ' + dfcntr['DEV'].iloc[0] + ' Mask: ' + dfcntr['RETICLE'].iloc[0]  + ' ERMSE: ' + str(round(min_rmse, 2)) + 'nm (' + str(min_pd) + ')'
    fig.update_layout(title={'text': title, 'font': {'size': 12}}, template=chart_theme)
    fig.update_xaxes(title_text='Xmm')
    fig.update_yaxes(title_text='Ymm')
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
    app.run_server(debug=True)
