import os
import sys
import csv
import pandas as pd
import plotly.graph_objects as go

from io import StringIO
from fastapi import FastAPI, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse

from modelo import Model
import htmlCont as hc

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)


app = FastAPI()

# Templates
templates = Jinja2Templates(directory="templates")

# Controller
filename = 'assets/reviews_result.csv'

@app.get('/')
async def root(request: Request):
    return templates.TemplateResponse("pagina.html", {"request": request})

@app.post("/predict-file")
async def predict_from_file(request: Request, file: UploadFile):
    print("Entro file")
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

    # create a dataframe from the numpy array
    sentimientos = pd.DataFrame(predictions)

    # replace values of 1 and 0 with "negativo" and "positivo", respectively
    sentimientos = sentimientos.replace({1: "negativo", 0: "positivo"})

    # rename the column to "sentimientos"
    sentimientos.columns = ["sentimiento"]

    # join the new dataframe to the existing dataframe "df"
    results_df = pd.concat([df, sentimientos], axis=1)
    # Save the results DataFrame to a CSV file in the "assets" folder
    results_df.to_csv(filename, index=False)

    # Return the HTML response to the user
    return templates.TemplateResponse('pagina.html', {"request": request, "prediction2": ['exitosa']})


@app.get("/predict-one-text")
async def predict_one_text(request: Request, input_text: str):
    print(input_text)
    model = Model()
    prediction = model.make_predictions_one(input_text)
    print(prediction)

    if prediction == 1:
        result = ["Postivio"]
    else:
        result = ["Negativo"]

    return templates.TemplateResponse('pagina.html', {"request": request, "prediction": result})


@app.get("/download")
async def get_data():
    return FileResponse(filename, filename="assets/reviews_result.csv")