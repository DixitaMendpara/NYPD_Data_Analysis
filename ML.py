import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load dataset
df = pd.read_csv('NYPD_Complaint_Data_Current__Year_To_Date_.csv')

# Selecting features and target variables
features = ['OFNS_DESC', 'BORO_NM', 'VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX']
targets = ['SUSP_RACE', 'SUSP_AGE_GROUP', 'SUSP_SEX']

# Encoding categorical data
label_encoders = {}
for col in features + targets:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Splitting the data
X = df[features]
Y = df[targets]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=500),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200)
}

# Training and testing each model
for name, model in models.items():
    print(f"\n{name}:")
    for target in targets:
        # Training
        model.fit(X_train, Y_train[target])

        # Prediction
        predictions = model.predict(X_test)

        # Calculating accuracy
        accuracy = accuracy_score(Y_test[target], predictions)
        print(f"Accuracy for predicting {target}: {accuracy:.2f}")

