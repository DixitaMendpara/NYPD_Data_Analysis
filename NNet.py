import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Load dataset
df = pd.read_csv('NYPD_Complaint_Data_Current__Year_To_Date_.csv')

# Selecting features
features = ['OFNS_DESC', 'BORO_NM', 'VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX']
targets = ['SUSP_SEX', 'SUSP_RACE', 'SUSP_AGE_GROUP']

# Encoding categorical data
for col in features + targets:
    df[col] = LabelEncoder().fit_transform(df[col])

# Splitting the data
X = df[features]
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Normalizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encoding the targets
Y = {}
for target in targets:
    Y[target] = OneHotEncoder(sparse_output=False).fit_transform(df[[target]])
    Y[target + '_train'], Y[target + '_test'] = train_test_split(Y[target], test_size=0.2, random_state=42)

# Neural network
input_layer = Input(shape=(X_train.shape[1],))

hidden = Dense(64, kernel_regularizer=l2(0.001))(input_layer)
hidden = BatchNormalization()(hidden)
hidden = Activation('relu')(hidden)
hidden = Dropout(0.1)(hidden)

hidden = Dense(128, kernel_regularizer=l2(0.001))(hidden)
hidden = BatchNormalization()(hidden)
hidden = Activation('relu')(hidden)
hidden = Dropout(0.2)(hidden)

# Output layers for each target
output_layers = [Dense(Y[target].shape[1], activation='softmax', kernel_regularizer=l2(0.001))(hidden) for target in targets]

model = Model(inputs=input_layer, outputs=output_layers)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss=['categorical_crossentropy']*len(targets),
              metrics=['accuracy'])

# Train the model
model.fit(X_train, [Y[target + '_train'] for target in targets], epochs=20, batch_size=64, validation_data=(X_test, [Y[target + '_test'] for target in targets]))

# Evaluate the model
losses, dense_2_loss, dense_3_loss, dense_4_loss, dense_2_accuracy, dense_3_accuracy, dense_4_accuracy = model.evaluate(X_test, [Y['SUSP_SEX_test'], Y['SUSP_RACE_test'], Y['SUSP_AGE_GROUP_test']])
print(f"Accuracy for predicting SUSP_SEX: {dense_2_accuracy:.2f}")
print(f"Accuracy for predicting SUSP_RACE: {dense_3_accuracy:.2f}")
print(f"Accuracy for predicting SUSP_AGE_GROUP: {dense_4_accuracy:.2f}")
