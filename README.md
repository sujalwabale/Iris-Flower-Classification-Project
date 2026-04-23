# Iris-Flower-Classification-Project
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print(df.shape)       # (150, 5)
print(df.head())
print(df['species'].value_counts())
