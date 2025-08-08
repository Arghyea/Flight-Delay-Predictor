
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


data = {
    'Month': [1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
    'DayofMonth': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'DayOfWeek': [1, 2, 3, 4, 5, 6, 7, 1, 2, 3],
    'CRSDepTime': [700, 945, 1300, 1600, 900, 1100, 1700, 800, 1400, 1900],
    'Distance': [500, 700, 300, 800, 1200, 600, 1000, 550, 650, 700],
    'DepDelay': [0, 5, 20, 35, -3, 10, 50, 0, 8, 25]
}

df = pd.DataFrame(data)

df['Delayed'] = df['DepDelay'].apply(lambda x: 1 if x > 15 else 0)

features = ['Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'Distance']
X = df[features]
y = df['Delayed']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

#Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Delayed')
plt.title("Count of Delayed vs Not Delayed Flights")
plt.tight_layout()
plt.savefig("delay_counts.png")

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Delayed', y='DepDelay')
plt.title("Departure Delay Distribution by Class")
plt.tight_layout()
plt.savefig("depdelay_boxplot.png")
