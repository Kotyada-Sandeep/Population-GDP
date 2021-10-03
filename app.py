import dash
import dash_bootstrap_components as dbc

external_stylesheets = [dbc.themes.BOOTSTRAP]

#external_stylesheets = ['https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)
app.title = 'Leads Generation & Sales Dashboard'
server = app.server
