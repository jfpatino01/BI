import os
import sys
import csv
import pandas as pd
import plotly.graph_objects as go

from io import StringIO
from fastapi import FastAPI, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse

from modelo import Model
import htmlCont as hc

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)


app = FastAPI()

# Templates
templates = Jinja2Templates(directory="templates")

# Controller
filename = 'uploaded/reviews_result.csv'

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("pagina.html", {"request": request})

@app.post("/predict-file")
async def predict_from_file(request: Request, file: UploadFile):
    # Check if uploaded file is CSV
    ext = os.path.splitext(file.filename)[1]
    if ext.lower() not in ['.csv']:
        return "Error: Only CSV files allowed"

    # Read the CSV file into a Pandas DataFrame
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode()))

    # Make predictions using the machine learning model
    model = Model()
    predictions = model.make_predictions(df)

    # Combine the input CSV and the predictions into a single DataFrame
    results_df = pd.concat([df, predictions['sentimiento'].replace({1: "negativo", 0: "positivo"})], axis=1)

    # Save the results DataFrame to a CSV file in the "assets" folder
    results_df.to_csv(filename, index=False)

    # Create a pie chart to visualize the sentiment distribution
    fig = go.Figure(
        data=[go.Pie(
            labels=results_df['sentimiento'].replace({1: "negativo", 0: "positivo"}).value_counts().index,
            values=results_df['sentimiento'].replace({1: "negativo", 0: "positivo"}).value_counts().values,
        )],
        layout=go.Layout(width=400, height=400),
    )
    fig.update_layout(
        title="Pie Chart: Sentiment of Reviews",
        legend_title="Sentiment",
    )

    # Generate HTML content for the pie chart
    html_content = hc.html_content_pie_graph() % fig.to_json()

    # Return the HTML response to the user
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/download")
async def get_data():
    # Return the results file as a download to the user
    return FileResponse(filename, filename="reviews_result.csv")