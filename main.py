import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

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

    if nb_values == 2:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

        # Feature Scaling by standardizing data (moyenne=0 et déviation standard=1)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # MLP binary
        model = keras.Sequential()
        model.add(tf.keras.layers.Input(shape=X_train[0].shape))
        model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(32, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(16, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))
        model.add(tf.keras.layers.Flatten())

        model.compile(tf.optimizers.Adam(), loss="binary_crossentropy", metrics=['accuracy'])
        history = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test))

        train_mse = model.evaluate(X_train, Y_train, verbose=1)
        test_mse = model.evaluate(X_test, Y_test, verbose=1)
        st.write(train_mse)
        st.write(test_mse)

        Y_pred = model.predict(X_test)
        Y_pred = Y_pred.flatten()
        Y_pred = np.rint(Y_pred)

        ax = plt.subplot()
        cm = confusion_matrix(Y_test, Y_pred)
        cm_df = pd.DataFrame(cm,
                             index=values,
                             columns=values)
        disp = sns.heatmap(cm_df, annot=True, fmt='', cmap='Blues')
        ax.set_xlabel('Predicted labels');
        ax.set_ylabel('True labels');
        disp.plot()
        st.pyplot()


    elif nb_values > 2:
        encoder = OneHotEncoder()
        Y = encoder.fit_transform(Y[:, np.newaxis]).toarray()

        # Train test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, stratify=Y, random_state=2)

        # Feature Scaling from -2 to 2
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # #MLP multi class
        model = keras.Sequential()
        model.add(tf.keras.layers.Input(shape=X_train[0].shape[0]))
        # model.add(tf.keras.layers.Dense(256, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(32, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(nb_values, activation=tf.keras.activations.softmax))
        model.add(tf.keras.layers.Flatten())

        model.compile(tf.optimizers.Adam(), loss="categorical_crossentropy", metrics=['accuracy'])

        history = model.fit(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test))

        train_mse = model.evaluate(X_train, Y_train, verbose=1)
        test_mse = model.evaluate(X_test, Y_test, verbose=1)

        st.write(train_mse)
        st.write(test_mse)

        Y_test = np.argmax(Y_test, axis=1)
        Y_pred = np.argmax(model.predict(X_test), axis=1)

        ax = plt.subplot()
        cm = confusion_matrix(Y_test, Y_pred)
        cm_df = pd.DataFrame(cm,
                             index=values,
                             columns=values)
        disp = sns.heatmap(cm_df, annot=True, fmt='', cmap='Blues')
        ax.set_xlabel('Predicted labels');
        ax.set_ylabel('True labels');
        disp.plot()
        st.pyplot()

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


def regression(df, in_var, out_var):
    in_var = int(in_var)-1
    out_var = int(out_var)-1
    dataset = df.values

    st.write(in_var)
    st.write(out_var)

    X = dataset[:, in_var, np.newaxis]
    Y = dataset[:, out_var]

    # X = dataset[:, 0:-1]
    # Y = dataset[:, -1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    model = linear_model.LinearRegression()
    model.fit(X, Y)

    Y_pred = model.predict(X)

    rms = mean_squared_error(Y, Y_pred, squared=False)
    st.write('Valeur moyenne :' + str(Y.mean()))
    st.write('Root Mean Square Error : ' + str(rms))

    plt.scatter(X, Y, color='black')
    # plt.scatter(X[:, 0], Y, color='black')
    plt.plot(X, Y_pred, color='red', linewidth=2)
    plt.xticks(())
    plt.yticks(())
    st.pyplot()

    matrice_corr = df.corr().round(1)
    disp = sns.heatmap(data=matrice_corr, annot=True)
    disp.plot()
    st.pyplot()

def main():
    with st.form(key='CSV specifics'):
        header = st.radio('Votre CSV comporte-t-il des headers ?', ('Oui', 'Non'))
        if header == 'Oui':
            HEAD = 0
        elif header == 'Non':
            HEAD = None

        data_file = st.file_uploader("Upload CSV", type=['csv'])
        if data_file is not None:
            file_details = {"Filename": data_file.name, "FileType": data_file.type, "FileSize": data_file.size}
            st.write(file_details)
            in_data = pd.read_csv(data_file, header=HEAD)
            in_data = pd.DataFrame(in_data)
            st.dataframe(in_data)
            st.write('Total rows : ' + str(len(in_data)))

        select_ml = st.radio('Quel type de modèle voulez-vous ?', ('Regression', 'Classification'))
        if 'Regression' in select_ml:
            in_var = st.text_input("Si régression, indiquez le numéro de colonne de votre variable d'entrée: ")
            out_var = st.text_input("Si régression, indiquez le numéro de colonne de votre variable de sortie: ")
        submit_type_ml = st.form_submit_button(label='Envoyer')

    if submit_type_ml:
        if select_ml == 'Regression':
            st.write('Regression')
            regression(in_data, in_var, out_var)

        elif select_ml == 'Classification':
            st.write('Classification')
            classification(in_data)


if __name__ == '__main__':
    main()
