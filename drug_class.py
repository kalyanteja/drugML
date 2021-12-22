import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
import pickle

df_drug = pd.read_csv("drug200.csv")

label_encoder = LabelEncoder()

categorical_features = [feature for feature in df_drug.columns if df_drug[feature].dtypes == 'O']
for feature in categorical_features:
    df_drug[feature]=label_encoder.fit_transform(df_drug[feature])
    
X = df_drug.drop("Drug", axis=1)
y = df_drug["Drug"]

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

kfold = KFold(random_state=42, shuffle=True)
cv_results = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
print(cv_results.mean(), cv_results.std())

with open('model_pickle', 'wb') as f:
    pickle.dump(model, f)

# reload model from Pickle
with open('model_pickle', 'rb') as f1:
    new_model = pickle.load(f1)

cv_results_new = cross_val_score(new_model, X, y, cv=kfold, scoring="accuracy")
print(cv_results_new.mean(), cv_results_new.std())

# new model is same as the old model output
assert cv_results_new.mean() == cv_results.mean()
assert cv_results_new.std() == cv_results.std()
