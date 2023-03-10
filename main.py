import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")


def get_inputs_titanic(df):
    df["Sex"] = pd.factorize(df["Sex"])[0]
    df["Embarked"] = pd.factorize(df["Embarked"])[0]

    sex = np.asarray(df["Sex"]).reshape(-1, 1).astype(np.float64)
    pclass = np.asarray(df["Pclass"]).reshape(-1, 1).astype(np.float64)
    pclass_1 = (pclass == 1).astype(np.float64)
    pclass_2 = (pclass == 1).astype(np.float64)
    pclass_3 = (pclass == 1).astype(np.float64)
    
    age = np.asarray(df["Age"]).reshape(-1, 1).astype(np.float64)
    age = np.nan_to_num(age, nan=-1)
    siblings_spouses = np.asarray(df["SibSp"]).reshape(-1, 1).astype(np.float64)
    parents_children = np.asarray(df["Parch"]).reshape(-1, 1).astype(np.float64)
    fare = np.asarray(df["Fare"]).reshape(-1, 1).astype(np.float64)
    embarked = np.asarray(df["Embarked"]).reshape(-1, 1).astype(np.float64)
    embarked_0 = (embarked == 0).astype(np.float64)
    embarked_1 = (embarked == 1).astype(np.float64)
    embarked_2 = (embarked == 2).astype(np.float64)
    
    cabin = np.asarray(df["Cabin"])
    cabin_known = (cabin.astype(str) != "nan").reshape(-1, 1)
    
    name = np.asarray(df["Name"])
    
    nickname = np.array([])
    for n in name:
        nickname = np.append(nickname, "(" in n)
    nickname = nickname.astype(np.float64).reshape(-1, 1)
    
    single_sets = (sex, age, siblings_spouses, parents_children, fare, embarked_0, embarked_1, embarked_2, pclass_1, pclass_2, pclass_3, cabin_known, nickname)
    preprocessed_data = np.concatenate(single_sets, axis=1)
    
    title = np.array([]).reshape(-1, 1)
    for n in name:
        start = n.index(",")
        end = n.index(".")
        title = np.append(title, n[start+2:end])
    title = title.reshape(-1, 1)

    for title_desc in ['Master', 'Mme', 'Don', 'Dr', 'Rev', 'Jonkheer', 'Capt', 'Miss', 'Col', 'Sir', 'Major', 'Mrs', 'Lady', 'Mr', 'Mlle', 'the Countess', 'Ms', 'Dona']:
        current_title = (title == title_desc).astype(np.float64)
        preprocessed_data = np.concatenate((preprocessed_data, current_title), axis=1)
    
    return preprocessed_data

def get_outputs_titanic(df):
    Y = df["Survived"]
    Y = np.asarray(Y).reshape(-1, 1).astype(np.float64)
    return Y

X_train = get_inputs_titanic(df)
Y_train = get_outputs_titanic(df)

model = Sequential([
    Dense(units=16, input_shape=(X_train.shape[1],), activation="relu"),
    Dense(units=32, activation="relu"),
    Dense(units=64, activation="relu"),
    Dropout(0.5),
    Dense(units=128, activation="relu"),
    Dense(units=256, activation="relu"),
    Dense(units=512, activation="relu"),
    Dense(units=2, activation="softmax")
    ])

model.summary()

model.compile(optimizer=Adam(lr=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

es_callback = EarlyStopping(monitor="val_loss", patience=80, restore_best_weights=True, baseline=0.7)
history = model.fit(x=X_train, y=Y_train, validation_split=0.4, batch_size=10, epochs=1000, shuffle=True, verbose=2, callbacks=[es_callback])
print(f'Trained for {len(history.history["loss"])} epochs.')


history = history.history
fig, ax = plt.subplots()
for key in ["loss", "val_loss"]:
    ax.plot(history[key], label=key)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
plt.title("Loss")
plt.show()

fig, ax = plt.subplots()
for key in ["accuracy", "val_accuracy"]:
    ax.plot(history[key], label=key)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
plt.title("Accuracy")
plt.show()


model.save("titanic_model.h5")

val_accuracy = history['val_accuracy'][-1]
print(f"Validation accuracy: {round(val_accuracy*100)}%")

def save_result():
    pred_df = pd.read_csv("test.csv")
    X_test = get_inputs_titanic(pred_df)
    
    predictions = model.predict_classes(X_test).reshape(-1, 1)
    
    passenger_id = pred_df["PassengerId"].to_numpy().reshape(-1, 1)
    solution = np.concatenate((passenger_id, predictions), axis=1)
    solution_df = pd.DataFrame(solution, columns=["PassengerId", "Survived"])
    solution_df.to_csv("submission.csv", index=False)
