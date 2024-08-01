# RAT
import pandas as pd

df = pd.read_csv("/content/RTA_Dataset.csv")
df.head(
df.info()
df.isnull().sum()
print(df['Accident_severity'].value_counts())
df['Accident_severity'].value_counts().plot(kind='bar')
df['Educational_level'].value_counts().plot(kind='bar')
dabl.plot(df, target_col='Accident_severity')
plt.figure(figsize=(6,5))
sns.countplot(x='Road_surface_type', hue='Accident_severity', data=df)
plt.xlabel('Rode surafce type')
plt.xticks(rotation=60)
plt.show
df['Time'] = pd.to_datetime(df['Time'])
new_df = df.copy()
new_df['Hour_of_Day'] = new_df['Time'].dt.hour
n_df = new_df.drop('Time', axis=1)
n_df.head()
features = ['Time','Day_of_week','Age_band_of_driver','Sex_of_driver','Educational_level','Type_of_vehicle','Cause_of_accident','Accident_severity' ]
target = n_df['Accident_severity']
feature_df = featureset_df.copy()
feature_df['Day_of_week'] = feature_df['Day_of_week'].fillna('Unknown')
feature_df['Age_band_of_driver'] = feature_df['Age_band_of_driver'].fillna('Unknown')
feature_df['Sex_of_driver'] = feature_df['Sex_of_driver'].fillna('unknown')
feature_df['Educational_level'] = feature_df['Educational_level'].fillna('Other')
feature_df['Type_of_vehicle'] = feature_df['Type_of_vehicle'].fillna('Unknown')
feature_df['Educational_level'] = feature_df['Educational_level'].fillna('Unknown')
feature_df['Cause_of_accident'] = feature_df['Cause_of_accident'].fillna('Unknown')
feature_df.info()
mport numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
from sklearn.datasets
import make_classification
from sklearn.model_selection
import train_test_split
from sklearn.metrics
import accuracy_score
from sklearn.linear_model
import LogisticRegression
nb_samples = 1000
x, y = make_classification(n_samples=nb_samples, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(xtrain, ytrain)
indices_binary = (y_train == 0) | (y_train == 1)
X_binary = X_train[indices_binary]
y_binary = y_train[indices_binary]

indices_test_binary = (y_test == 0) | (y_test == 1)
X_test_binary = X_test[indices_test_binary]
y_test_binary = y_test[indices_test_binary]

classifier = LinearSVC()
classifier.fit(X_binary, y_binary)

display = PrecisionRecallDisplay.from_estimator(
    classifier, X_test_binary, y_test_binary, name="LinearSVC", plot_chance_level=True
)
_ = display.ax_.set_title("RAT")
