import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("my_dataset.csv")

# Scatterplot with year as point size
sns.scatterplot(data=df, x="horsepower", y="speed", size="year", hue="model", legend=False)
plt.title("Custom Dataset - Horsepower vs Speed (Size=Year)")
plt.xlabel("Horsepower")
plt.ylabel("Speed")
plt.show()

# Additional Visualization 1: Pairplot of features colored by model
sns.pairplot(df, vars=["horsepower", "speed", "year"], hue="model", palette="tab10")
plt.suptitle("Pairplot of Horsepower, Speed, and Year by Model", y=1.02)
plt.show()

# Additional Visualization 2: Boxplot of Speed distribution per Model
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="model", y="speed", palette="Set2")
plt.title("Speed Distribution per Model")
plt.xlabel("Model")
plt.ylabel("Speed")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Prepare data for training
X = df[["horsepower", "speed", "year"]]
le = LabelEncoder()
y = le.fit_transform(df["model"])

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("✅ Model trained and saved.")
