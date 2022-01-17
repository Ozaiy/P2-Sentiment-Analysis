import dash
import json
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from pymongo import MongoClient
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import nltk
from nltk import FreqDist
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud,STOPWORDS
import numpy as np

client = MongoClient("localhost:27017")

aggregation = [
    {
        '$group': {
            '_id': {
                'Hotel_Name': '$Hotel_Name', 
                'Average_Score': '$Average_Score', 
                'Hotel_Address': '$Hotel_Address', 
                'lat': '$lat', 
                'lng': '$lng',
            }, 
            'dups': {
                '$addToSet': '$_id'
            }, 
            'count': {
                '$sum': 1
            }
        }
    }, {
        '$project': {
            '_id': 0, 
            'Hotel_Name': '$_id.Hotel_Name', 
            'Average_Score': '$_id.Average_Score', 
            'Hotel_Address': '$_id.Hotel_Address', 
            'lat': '$_id.lat', 
            'lng': '$_id.lng',
        }
    }
]



db=client.indv
result=db.reviews.find({})
source=list(result)
df = pd.DataFrame(source)

mapDf = db.reviews.aggregate(aggregation)
mapDf = list(mapDf)
mapDf = pd.DataFrame(mapDf)

nnmatrix = pd.DataFrame(list(db.nnmatrix.find({})))
nn_cf_matrix = []
nn_cf_matrix.append(nnmatrix['true_neg'].array[0])
nn_cf_matrix.append(nnmatrix['false_pos'].array[0])
nn_cf_matrix.append(nnmatrix['false_neg'].array[0])
nn_cf_matrix.append(nnmatrix['true_pas'].array[0])
nn_cf_matrix = np.array(nn_cf_matrix)
nn_cf_matrix = nn_cf_matrix.reshape(2,2)

nn_heat_plot = dcc.Graph(figure=px.imshow(nn_cf_matrix))

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "14rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "box-shadow": "0 4px 8px 0 rgba(0,0,0,0.2)",
    "transition": "0.3s",
    "box-shadow": "0 8px 16px",
    "width": "15%"
}

SIDEBAR_STYLE_STEDEN = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "13%",
    "padding": "1%",
    "margin-left": "17%",
    "margin-top": "2%",
    "margin-bottom": "33%",
    "background-color": "#f8f9fa",
    "box-shadow": "0 4px 8px 0 rgba(0,0,0,0.2)",
    "transition": "0.3s",
    "box-shadow": "0 8px 16px",
    "border-radius": "8px",
}

SIDEBAR_STYLE_HOVER = {

    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "13%",
    "padding": "1%",
    "margin-left": "17%",
    "margin-top": "20%",
    "margin-bottom": "1%",
    # "margin-right": "80%",
    "background-color": "#f8f9fa",
    "box-shadow": "0 4px 8px 0 rgba(0,0,0,0.2)",
    "transition": "0.3s",
    "box-shadow": "0 8px 16px",
    "border-radius": "8px",
}

MAP_THIRD_CHARTS_STYLE = {
    "margin-top": "1%",
    'display': 'inline-block',
    "box-shadow": "0 4px 8px 0 rgba(0,0,0,0.2)",
    "transition": "0.3s",
    "box-shadow": "0 8px 16px",
    "border-radius": "8px",
    "margin-left": "20%",
    "width": "35%",
    "margin-bottom": "1%",
}

MAP_SECOND_CHARTS_STYLE = {
    "margin-top": "1%",
    'display': 'inline-block',
    "box-shadow": "0 4px 8px 0 rgba(0,0,0,0.2)",
    "transition": "0.3s",
    "box-shadow": "0 8px 16px",
    "border-radius": "8px",
    "margin-left": "4%",
    "width": "35%",
    "margin-bottom": "1%",
}

MAP_FIRST_CHART_STYLE ={
    "margin-left": "20%",
    "margin-top": "1%",
    'display': 'inline-block',
    'width': '38%',
    "box-shadow": "0 4px 8px 0 rgba(0,0,0,0.2)",
    "transition": "0.3s",
    "box-shadow": "0 8px 16px",
    "border-radius": "8px",
    "width": "35%",
    "margin-bottom": "1%",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "20rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

MAP_STYLE = {
    "box-shadow": "0 4px 8px 0 rgba(0,0,0,0.2)",
    "transition": "0.3s",
    "box-shadow": "0 8px 16px",
    "border-radius": "12px",
    "margin-left": "20%",
    "margin-bottom": "1%",
    
}

content = html.Div(id="page-content", style=CONTENT_STYLE, children=[])

# plot voor woorden
def plot_wordBar(x, title_text):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
    d = words_df.nlargest(columns="count", n = 15) 
    plt.figure(figsize=(20,5))
    ax = px.bar(d,x= "word", y = "count", title=title_text, height=350)
    return ax
    

# main layout van de app
app.layout = html.Div([

    # keuze menu
    html.Div(id='sidebar', children=[

        html.H2('Dashboard'),
        html.Hr(),
        html.H3('Ozcan Isik 500800699'),

        html.Hr(),
        
        dcc.Location(id="url"),
        dbc.Nav(
            [
                dbc.NavLink("Map", href="/", active="exact"),
                html.Br(),
                dbc.NavLink("Reviews", href="/page-1", active="exact"),
                html.Br(),
                dbc.NavLink("Benchmark", href="/page-2", active="exact"),
            ],
            vertical=True,
        ),

        
        ], style=SIDEBAR_STYLE),

    # map
    html.Div(id="page-content", children=[], style=CONTENT_STYLE),
    
    ])

# /////////////////////////
# /////Pagina Contents/////
# /////////////////////////

# pagina 3 benchmark models
page_3_content = html.Div([
    html.H1('Benchmarks van neurale netwerken'),
    dcc.RadioItems(
        id='page-2-radios',
        options=[{'label': i, 'value': i} for i in ['Orange', 'Blue', 'Red']],
        value='Orange'
    ),
    html.Div(id='page-3-content'),
    nn_heat_plot
])

# pagina 2 reviews
page_2_content = html.Div([
    html.H1('Inzicht In Reviews'),
    dcc.RadioItems(
        id='page-2-radios',
        options=[{'label': i, 'value': i} for i in ['Orange', 'Blue', 'Red']],
        value='Orange'
    ),
    html.Div(id='page-2-content'),
])

# pagina 1 maps
page_1_content = html.Div([
    # html.H1('Map van hotels', style={"margin-right": "55rem"}),
    html.Div(
        children=[
        html.H3('Steden'),

        dcc.RadioItems(
        id="page-1-radios",
        options=[{'label': i, 'value': i} for i in ['Amsterdam', 'London', 'Paris', 'Barcalona', 'Vienna', 'Milano']],
        value="Amsterdam",
        style={"margin-right": "55%"})
        
        ],
        style=SIDEBAR_STYLE_STEDEN
     ),

        #    map is hier geplaats
    html.Div([html.H4('Hover over een data punt op de map')], style={"margin-left": "20%"}),
    html.Div(id='page-1-content', style=MAP_STYLE),
    html.Div([dcc.Markdown(), html.Div(id='hover-data')]),

    ])

# callbacks voor MAP pagina 
@app.callback(Output("page-1-content", "children"), [Input("page-1-radios", "value")])
def change_coords(value):

    if value == "Amsterdam":
        coords = [52.36,4.89]
    elif value == "London":
        coords = [51.5085300,-0.1257400]
    elif value == "Paris":
        coords = [48.8534100,2.3488000]
    elif value == "Barcalona":
        coords = [41.3887900,2.1589900]
    elif value == "Vienna":
        coords = [48.2084900,16.3720800]
    elif value == "Milano":
        coords = [45.4642700,9.1895100]


    # Plotting van de MAP
    fig = go.Figure(data=px.scatter_mapbox(mapDf,
    lat = 'lat',
    lon = 'lng',
    hover_name='Hotel_Name',
    hover_data=['Average_Score', 'Hotel_Address'],
    zoom=12, 
    center={"lat": coords[0], "lon": coords[1]},
    color='Average_Score',
    height=300,
    ))

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_layout(clickmode='event+select')
    # fig.update_traces(marker_size=7)

    return dcc.Graph(id='map',figure=fig)

# pagina menu
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return page_1_content
    elif pathname == "/page-1":
        return page_2_content
    elif pathname == "/page-2":
        return page_3_content
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

# callbacks for page 3
@app.callback(Output('page-3-content', 'children'),
              [Input('page-3-radios', 'value')])
def page_3_radios(value):
    # nnmatrix = pd.DataFrame(list(db.nnmatrix.find({})))
    # nnhistory = pd.DataFrame(list(db.nnhistory.find({})))



    return html.Div([html.H1('zuwed')])

# callbacks for page 2
@app.callback(Output('page-2-content', 'children'),
              [Input('page-2-radios', 'value')])
def page_2_radios(value):

    shortdf = df[:10000]
    plot_pos = plot_wordBar(shortdf['Positive_Review']
    , 'Meest voorkomende woorden in Positive Reviews')

    plot_neg = plot_wordBar(shortdf['Negative_Review']
    , 'Meest voorkomende woorden in Negative Reviews')

    review_df = df.set_index('Review_Date')

    review_df.index = pd.to_datetime(review_df.index)

    # review_df.sort_values(by='Review_Date', inplace=True)

    gem_per_land = review_df.groupby('Reviewer_Nationality').mean()

    gem_negatives = px.line(gem_per_land, y='Review_Total_Negative_Word_Counts',
    title='Gemiddelde aantal negative woorden per land')

    gem_pos = px.line(gem_per_land, y='Review_Total_Positive_Word_Counts',
    title='Gemiddelde aantal positive woorden per land')

    gem_score = px.line(gem_per_land, y='Reviewer_Score',
    title='Gemiddelde review score per land')

    return html.Div([
        dcc.Graph(figure=plot_pos),
        dcc.Graph(figure=plot_neg),
        dcc.Graph(figure=gem_negatives),
        dcc.Graph(figure=gem_pos),
        dcc.Graph(figure=gem_score),])


# hoort bij page 1 hovering
@app.callback(
    Output('hover-data', 'children'),
    Input('map', 'hoverData'))
def display_hover_data(hoverData):

    locatie = html.Div([
        html.Hr(),
        html.H4('Locatie'),
        html.P("Adres: " + hoverData['points'][0]['customdata'][1]),
        html.P("Latitude: " + str(hoverData['points'][0]['lat'])),
        html.P("Longtitude: " + str(hoverData['points'][0]['lon'])),
    ])

    current_hotel = html.Div([
        # html.H4("Hotel"),
        html.H5(hoverData['points'][0]['hovertext']),
        html.Hr()
    ])

    average_score = html.Div([
        html.H4("Gemidelde Score"),
        html.H5(str(hoverData['points'][0]['customdata'][0]), style={"color": "rgb(235, 180, 63)"})
    ])

    querydata = db.reviews.find({"Hotel_Name": hoverData['points'][0]['hovertext']}) 

    current_hoteldf = pd.DataFrame(list(querydata))

    current_hoteldf = current_hoteldf.set_index('Review_Date')
    current_hoteldf.index = pd.to_datetime(current_hoteldf.index)
    current_hoteldf.sort_values(by='Review_Date', inplace=True)

    total_reviews = html.Div([
    html.H4('Totale Reviews'),
    html.H5(current_hoteldf['Total_Number_of_Reviews'][0],style={"color": "blue"})
    ])

    hist_nationality = px.histogram(current_hoteldf, x='Reviewer_Nationality',
    title='Aantal Reviewers Per Nationaliteit')

    gem_score_per_land = current_hoteldf.groupby('Reviewer_Nationality').mean()

    hist_score = px.line(gem_score_per_land, y='Reviewer_Score',
    title='Gemiddelde Review Score Per Nationaliteit')

    hover_pos = plot_wordBar(current_hoteldf['Positive_Review']
    , 'Meest voorkomende woorden in Positive Reviews')

    hover_neg = plot_wordBar(current_hoteldf['Negative_Review']
    , 'Meest voorkomende woorden in Negative Reviews')

    side_info = html.Div([current_hotel, average_score,total_reviews,locatie], style=SIDEBAR_STYLE_HOVER)

    charts = html.Div(children=[dcc.Graph(figure=hover_pos)],
     style=MAP_FIRST_CHART_STYLE)
    
    charts1 = html.Div(children=[dcc.Graph(figure=hover_neg)]
    ,style=MAP_SECOND_CHARTS_STYLE)

    charts2 = html.Div(children=[dcc.Graph(figure=hist_nationality)]
    ,style=MAP_THIRD_CHARTS_STYLE)

    charts3 = html.Div(children=[dcc.Graph(figure=hist_score)]
    ,style=MAP_SECOND_CHARTS_STYLE)

    return html.Div([side_info, charts, charts1, charts2, charts3])

app.run_server(debug=True)