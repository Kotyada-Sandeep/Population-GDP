import dash
import dash_table
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime as dte
import psycopg2
import pyodbc
import math
import flask
import socket
from app import app



dropdown_region_df = pd.read_csv("Docs/Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_3011530.csv")
dropdown_region_df = dropdown_region_df.fillna(value='Others')
region_df = pd.DataFrame(dropdown_region_df)
dropdown_region_options = dropdown_region_df["Region"].unique()
region_options_list = []
for region_options in dropdown_region_options:
    region_options_list.append({'label':region_options, 'value':region_options})

dropdown_country_options = dropdown_region_df["TableName"]

def dashtable(id_name,alt_row_color):
        datatable = dash_table.DataTable(id=id_name,export_format='csv',export_headers='display',sort_action="native",filter_action="native",
                                         page_action="native",page_current=0,
                                         page_size=10,                                       #style_as_list_view=True,
                                         style_header={'backgroundColor': 'rgb(230,230,230)',
                                                       'fontWeight': 'bold'},
                                         style_cell={'whiteSpace': 'normal',
                                                     'height': 'auto',
                                                     'textAlign': 'center'},
                                         style_cell_conditional=[{'if': {'column_id': c},
                                                                  'textAlign': 'left'} for c in [alt_row_color]],
                                         style_data_conditional=[{'if': {'row_index': 'odd'},
                                                                  'backgroundColor': 'rgb(248,248,248)'}])
        return datatable
    
    
master_db_layout = html.Div([html.Div([dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(dbc.NavbarBrand("World Population & GDP 1960 to 2020", style={'color':'black', 'fontWeight':'bold'})),
                ],
                align="center",
                no_gutters=True,
            ),
        ),
    ],
    color="#80808045",
    dark=True,
    fixed='top',
    sticky='sticky',
    style={'backgroundColor': '	#A0A0A0','position':'fixed', 'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)'},
    ),]),
                             ])



app.layout = html.Div([html.Div([
    master_db_layout]),
                       html.Br(),
                       html.Br(),
                       html.Br(),
                       html.Div([
                                 html.Div([
                                     html.P("Region"),
                                     dcc.Dropdown(
                    id='region-dropdown',
                    options=region_options_list,
                    value='South Asia',
                    clearable=False)], style={'width':'15%','textAlign':'center','color':'black','float':'left','padding':'20px','box-sizing':'border-box','fontWeight':'600'}
                                     #, className="dropdown-right-labels"
                                     ),
                                 html.Div([
                                     html.P("Country"),
                                     dcc.Dropdown(
                                     id = 'country-dropdown',
                                     #options = country_options_list,
                                     #value = 'India',
                                     clearable=False,
                                     multi=True)],style={'width':'15%','textAlign':'center','color':'black','float':'left','padding':'20px','box-sizing':'border-box','fontWeight':'600'}
                            ),
                                 html.Div([html.Button(
                         id='submit_button',
                         n_clicks=0,children='Submit',style={'display':'inline-block', 'color': 'white', 'backgroundColor':'#119DFF', 'fontSize':18, 'marginLeft': '20px'}
                         )], style={'width':'15%','textAlign':'left','color':'black','float':'left','padding':'20px','box-sizing':'border-box','fontWeight':'600'}),],className="row"),
                       html.Br(),
                       html.Br(),
                       html.Div([dcc.Loading(id='summary_loading_icon4',children=[
                           html.P('Population of Countries from 1960 to 2020'),
                           dcc.Graph(id='graph')],type='default'),]),
                       html.Br(),
                       html.Br(),
                       html.Br(),
                       html.Br(),
                       html.Div([
                           html.Div(id='map_title', children=[]),
                           dcc.Graph(id='choropleth-map')
                           ]),
                       html.Div([
        dcc.Input(id='input_state', type='number', inputMode='numeric', value=2020,
                  max=2020, min=1960, step=5, required=True),
        html.Button(id='map_submit_button', n_clicks=0, children='Submit')
        ],style={'text-align': 'center'}),
                       html.Div([
                           dcc.Loading(id='summary_loading_icon1',children=[
                               html.Div(id='table_title', children=[]),
                               html.Div(dashtable('datatable1','Country Name'))],type='default')])
                       ])


@app.callback(
    Output('country-dropdown', 'options'),
    Output('country-dropdown', 'value'),
    Input('region-dropdown', 'value'))
def set_cities_options(selected_region):
    
    df = region_df[(region_df['Region'] == selected_region)]
    #country_options_list = []
    country_options_list = df['TableName'].unique()

    return [{'label': i, 'value': i} for i in country_options_list], [{'label': i, 'value': i} for i in country_options_list] 

@app.callback(
    Output('graph', 'figure'),
    [Input('submit_button', 'n_clicks')],
    [State('country-dropdown', 'value')]
    )
def graph(n_clicks, country_value):
    test_df = pd.read_csv("Docs/API_SP.POP.TOTL_DS2_en_csv_v2_3011530.csv")
    #print(test_df.head(10))
    #test_df = test_df[test_df['Country Name'] == 'Aruba']
    #print(test_df)
    del test_df['Country Code']
    del test_df['Indicator Name']
    del test_df['Indicator Code']
    df5=test_df.melt(id_vars=['Country Name'])
    df5.rename(columns = {'variable':'Year', 'value':'Population'}, inplace = True)

    #df6 = df5[(df5['Country Name'] == country_value)]
    if country_value:
        df5 = df5[df5['Country Name'].isin(country_value)]
    fig = px.bar(df5, x="Year", y="Population", 
                 color="Country Name", barmode="group")
    return fig

@app.callback(
    [Output('map_title', 'children'),
     Output('table_title', 'children'),
     Output('choropleth-map', 'figure'),
     Output('datatable1', 'data'),
     Output('datatable1', 'columns')],
    [Input('map_submit_button', 'n_clicks')],
    [State('input_state', 'value')]
    )

def choropleth_map(n_clicks, value_selected):
    population_df = pd.read_csv("Docs/API_SP.POP.TOTL_DS2_en_csv_v2_3011530.csv")
    del population_df['Country Code']
    del population_df['Indicator Name']
    del population_df['Indicator Code']
    population_df = population_df.melt(id_vars=['Country Name'])
    print(population_df)
    #print(df.info())
    population_df['variable'] = population_df['variable'].astype(int)
    #print(df.info())
    population_df.rename(columns = {'variable':'Year', 'value':'Population'}, inplace = True)
    print(population_df.dtypes)

    gdp_df = pd.read_csv("Docs/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_3011433.csv")
    gdp_df.fillna(value='N/A')
    del gdp_df['Country Code']
    del gdp_df['Indicator Name']
    del gdp_df['Indicator Code']
    gdp_df = gdp_df.melt(id_vars=['Country Name'])
    gdp_df.rename(columns = {'variable':'Year', 'value':'GDP'}, inplace = True)
    gdp_df['Year'] = gdp_df['Year'].astype(int)
    gdp_df['GDP'] = gdp_df['GDP'].astype(float)
    gdp_df['GDP'] = gdp_df['GDP']
    print(gdp_df.dtypes)
    print(gdp_df)
    df=pd.merge(population_df,gdp_df,how='left',left_on=['Country Name', 'Year'], right_on=['Country Name', 'Year'])
    df1 = df[df['Year']==value_selected]
    
    table_df1 = pd.DataFrame(df1, columns = df1.columns)
    table_df2 = table_df1.pivot_table(index=['Country Name'], values=['Population', 'GDP'],
                          aggfunc={'Population': 'sum', 'GDP': 'sum'},
                          margins=True, margins_name='Grand Total')
    # These are the custom calculated columns of df2

    table_df2['GDP per capita'] = ((table_df2['GDP']/table_df2['Population']).apply('${:0f}'.format))
    table_df2['GDP'] = table_df2['GDP'].apply('${:,}'.format)
    #print(table_df2.dtypes)
    table_df2['Population'] = table_df2['Population'].astype(int).apply(lambda x: f'{x:,}')
    table_df2.reset_index(inplace=True)
    table_df3 = table_df2[['Country Name', 'GDP', 'Population', 'GDP per capita']]
    
    fig = px.choropleth(df1, locations="Country Name", 
                    locationmode='country names', color="Population", 
                    hover_name="Country Name",
                    hover_data = ['Year', 'Population', 'GDP'],
                    color_continuous_scale="dense")
    fig.update(layout_coloraxis_showscale=True)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))

    return html.P('Population of Countries in {}'.format(value_selected)), html.P('GDP of Countries in {}'.format(value_selected)), fig, table_df3.to_dict('records'),[{"name": i, "id": i} for i in table_df3.columns]


if __name__ == "__main__":
                             app.run_server()
