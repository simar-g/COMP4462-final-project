from flask import Flask

flask_app=Flask(__name__)



from application import routes

# from routes import dash_app

# # Register the Dash app
# app = dash_app.server