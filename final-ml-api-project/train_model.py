# train_model.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("my_dataset.csv")

# Using seaborn scatterplot with year as point size (no 'z' argument)
sns.scatterplot(data=df, x="horsepower", y="speed", size="year", hue="model", legend=False)
plt.title("Custom Dataset")
plt.xlabel("Horsepower")
plt.ylabel("Speed")
plt.show()

X = df[["horsepower", "speed", "year"]]
le = LabelEncoder()
y = le.fit_transform(df["model"])

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("✅ Model trained and saved.")
