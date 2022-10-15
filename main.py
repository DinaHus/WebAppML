import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from tensorflow import keras
from keras import layers
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

st.write('''
Bienvenue dans la WebApp de prédiction CSV
''')

st.subheader("Send Dataset")


def classification(df):
    last_col = df.columns[-1]
    values = df[last_col].unique()
    nb_values = len(values)

    dataset = df.values

    X = dataset[:, 0:-1]
    Y = dataset[:, -1]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    model = keras.Sequential()
    model.add(tf.keras.layers.Input(shape=X_train[0].shape))

    # CNN
    model.add(layers.Conv1D(32, 2, activation='relu', padding='same'))
    model.add(layers.MaxPool1D())
    model.add(layers.Conv1D(64, 2, activation='relu', padding='same'))
    model.add(layers.MaxPool1D())
    model.add(layers.Conv1D(128, 2, activation='relu', padding='same'))
    model.add(layers.MaxPool1D())
    model.add(layers.Conv1D(128, 2, activation='relu', padding='same'))
    model.add(layers.Conv1D(128, 2, activation='relu', padding='same'))

    model.add(layers.Flatten())

    model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(32, activation=tf.keras.activations.relu))
    model.add(layers.Dense(nb_values - 1, activation='sigmoid'))
    model.compile(tf.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    st.pyplot()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    st.pyplot()

    Y_pred = model.predict(X_test)
    Y_pred = Y_pred.flatten()
    Y_pred = np.rint(Y_pred)
    cm = confusion_matrix(Y_test, Y_pred)
    labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    labels = np.asarray(labels).reshape(2, 2)
    disp = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
    disp.plot()
    st.pyplot()

def regression(df):
    last_col = df.columns[-1]
    values = df[last_col].unique()

    dataset = df.values

    X = dataset[:, 0:-1]
    Y = dataset[:, -1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    rms = mean_squared_error(Y_test, Y_pred, squared=False)
    st.write('Root Mean Square Error : ' + str(rms))

    plt.scatter(X_test, Y_test, color='black')
    plt.plot(X_test, Y_pred, color='red', linewidth=2)
    plt.xticks(())
    plt.yticks(())
    st.pyplot()

def main():
    with st.form(key='CSV specifics'):
        data_file = st.file_uploader("Upload CSV", type=['csv'])
        if data_file is not None:
            file_details = {"Filename": data_file.name, "FileType": data_file.type, "FileSize": data_file.size}
            st.write(file_details)
            in_data = pd.read_csv(data_file)
            in_data = pd.DataFrame(in_data)
            st.dataframe(in_data)
            st.write('Total rows : ' + str(len(in_data)))
        header = st.radio('Votre CSV comporte-t-il des headers ?', ('Oui', 'Non'))
        select_ml = st.radio('Quel type de modèle voulez-vous ?', ('Regression', 'Classification'))
        submit_type_ml = st.form_submit_button(label='Envoyer')

    if submit_type_ml:
        if select_ml == 'Regression':
            st.write('Regression')
            regression(in_data)

        elif select_ml == 'Classification':
            st.write('Classification')
            classification(in_data)

if __name__ == '__main__':
    main()