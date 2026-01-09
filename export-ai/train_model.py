import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Örnek veri (sonra büyütülecek)
data = {
    "urun_tipi": ["tekstil", "tekstil", "gida", "makine", "gida", "makine"],
    "ulke": ["Almanya", "Fransa", "Irak", "Almanya", "Rusya", "Italya"],
    "birim_fiyat": [12, 11, 5, 200, 4, 180],
    "teslim_sekli": ["FOB", "FOB", "EXW", "CIF", "EXW", "CIF"],
    "basarili": [1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Encode
encoders = {}
for col in ["urun_tipi", "ulke", "teslim_sekli"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("basarili", axis=1)
y = df["basarili"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Kaydet
joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("Model ve encoderlar kaydedildi")
