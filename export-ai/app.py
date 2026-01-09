from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="AI Export Assistant")

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

class ExportInput(BaseModel):
    urun_tipi: str
    ulke: str
    birim_fiyat: float
    teslim_sekli: str

@app.get("/")
def home():
    return {"status": "AI Export Assistant is running"}

@app.post("/tahmin")
def tahmin(data: ExportInput):
    df = pd.DataFrame([{
        "urun_tipi": encoders["urun_tipi"].transform([data.urun_tipi])[0],
        "ulke": encoders["ulke"].transform([data.ulke])[0],
        "birim_fiyat": data.birim_fiyat,
        "teslim_sekli": encoders["teslim_sekli"].transform([data.teslim_sekli])[0]
    }])

    sonuc = model.predict(df)[0]
    return {"basarili": bool(sonuc)}
