import joblib

model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

# Sample input with horsepower, speed, year (add a realistic year)
sample = [[150, 180, 2021]]

prediction = model.predict(sample)
print("Prediction:", le.inverse_transform(prediction)[0])
