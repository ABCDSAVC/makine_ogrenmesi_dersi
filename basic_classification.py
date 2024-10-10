from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class BasicClassification:
    def __init__(self):
        self.model = LogisticRegression()
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

data = {
    'GPA': [2.5, 3.0, 1.8, 3.5, 2.2, 3.8, 1.5, 2.0, 3.9, 2.8],  
    'Passed': [0, 1, 0, 1, 0, 1, 0, 0, 1, 1]  
}

df = pd.DataFrame(data)

X = df[['GPA']].values  
y = df['Passed'].values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Tahmin edilen sonuçlar:", y_pred)
print("Gerçek sonuçlar:", y_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Modelin doğruluğu: {accuracy * 100:.2f}%")

new_gpa = np.array([[3.5]]) 
predicted_pass = model.predict(new_gpa)
print(f"Not ortalaması {new_gpa} olan öğrenci tahmini: {'Geçti' if predicted_pass[0] == 1 else 'Kaldı'}")