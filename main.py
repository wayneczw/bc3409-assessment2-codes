import pandas as pd
import numpy as np
import gc
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
import collections
import operator
import matplotlib.pyplot as plt

from pandas.plotting import autocorrelation_plot

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures

from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional
from keras.layers import Convolution1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import GlobalMaxPool1D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import SpatialDropout1D
from keras.layers import concatenate
from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing import text


CAT_FEATURES = [
    'Day of Week',
    'Day of Month',
    'Month of Year']

CONT_FEATURES = [
    'Open', 'High', 'Low', 'Adj Close', 'Volume',
    'Crude Oil', 'Brent Crude',
    'Unleaded Gasoline', 'Heating Oil', 'Gas Oil', 'Natural Gas',
    'Crude Oil_MA', 'Brent Crude_MA', 'Unleaded Gasoline_MA',
    'Heating Oil_MA', 'Gas Oil_MA', 'Natural Gas_MA', 'Crude Oil_Change',
    'polarity', 'subjectivity',
    ]

DJI = [
    'Open', 'High', 'Low', 'Adj Close', 'Volume',
]

COM = [
    'Crude Oil', 'Brent Crude',
    'Unleaded Gasoline', 'Heating Oil', 'Gas Oil', 'Natural Gas',
]

COM_MA = [
    'Crude Oil_MA', 'Brent Crude_MA', 'Unleaded Gasoline_MA',
    'Heating Oil_MA', 'Gas Oil_MA', 'Natural Gas_MA',
]

TEXT = [
    'polarity', 'subjectivity',
]

TARGET_FEATURE = ['Crude Oil_Change']

seed = 2019
win_size = 25

batch_size = 4
epochs = 16

polynomial = False

np.random.seed(seed)
tf.set_random_seed(seed)

fig_path = 'trend_plot.png'


def build_model_vanilla(
    x_input_shape,
    output_shape,
    dropout_rate=0,
    kernel_regularizer=0,
    activity_regularizer=0,
    bias_regularizer=0,
    **kwargs):

    x_input = Input(x_input_shape, name='x_input')
    x = Flatten()(x_input)
    x = Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(
        16,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(
        8,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(x)
    x = Dropout(dropout_rate)(x)

    output = Dense(output_shape, activation='sigmoid', name='output')(x)

    model = Model(inputs=[x_input], outputs=[output])
    model.compile(optimizer=optimizers.Adam(0.0005, decay=1e-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
#end def


def build_model_gru(
    x_input_shape,
    output_shape,
    dropout_rate=0,
    kernel_regularizer=0,
    activity_regularizer=0,
    bias_regularizer=0,
    **kwargs):

    x_input = Input(x_input_shape, name='x_input')
    x = Bidirectional(GRU(16, return_sequences=True))(x_input)
    x = Convolution1D(8, 3, activation="relu")(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(
        8,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(
        4,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(x)
    x = Dropout(dropout_rate)(x)

    output = Dense(output_shape, activation='sigmoid', name='output')(x)

    model = Model(inputs=[x_input], outputs=[output])
    model.compile(optimizer=optimizers.Adam(0.0005, decay=1e-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
#end def


def build_model_lstm(
    x_input_shape,
    output_shape,
    dropout_rate=0,
    kernel_regularizer=0,
    activity_regularizer=0,
    bias_regularizer=0,
    **kwargs):

    x_input = Input(x_input_shape, name='x_input')
    x = Bidirectional(LSTM(16, return_sequences=True))(x_input)
    x = Convolution1D(8, 3, activation="relu")(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(
        8,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(
        4,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(x)
    x = Dropout(dropout_rate)(x)

    output = Dense(output_shape, activation='sigmoid', name='output')(x)

    model = Model(inputs=[x_input], outputs=[output])
    model.compile(optimizer=optimizers.Adam(0.0005, decay=1e-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
#end def


def prepare_data(
    df,
    target_feature,
    cat_features=None,
    cont_features=None,
    window=25
    ):

    if cat_features is not None:
        # encode categorical features
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        X_cat_raw = cat_encoder.fit_transform(df[cat_features].values)

    if cont_features is not None:
        X_cont_raw = df[cont_features].values
        if polynomial:
            poly = PolynomialFeatures()
            X_cont_raw = poly.fit_transform(X_cont_raw)

    if cat_features is not None and cont_features is not None:
        X_raw = np.hstack([X_cont_raw, X_cat_raw])
    elif cat_features is not None:
        X_raw = X_cat_raw
    elif cont_features is not None:
        X_raw = X_cont_raw

    Y_raw = df[target_feature].values

    ###### Prepare Data
    X = list()
    Y = list()
    for i in range(Y_raw.shape[0]):
        if i < window:
            continue

        _x = [X_raw[i-k] for k in range(window, 0, -1)]

        X.append(_x)
        Y.append(Y_raw[i][0])
    #end for

    X = np.asarray(X)
    Y = np.asarray(Y)

    return X, Y
#end def


def main():
    df = pd.read_csv('processed_data.csv')
    df = df.dropna()

    '''
    # the following chunk of code is for data visualisation
    tmp_df = df[:30]
    tmp_df.plot(x='Date', y=DJI+['Crude Oil'])
    plt.title("DJI VS Crude Oil")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0))
    plt.grid(b=True)
    plt.savefig('1_' + fig_path, bbox_inches='tight')

    plt.figure()
    tmp_df.plot(x='Date', y=COM+['Crude Oil'])
    plt.title("Commodities VS Crude Oil")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0))
    plt.grid(b=True)
    plt.savefig('2_' + fig_path, bbox_inches='tight')

    plt.figure()
    tmp_df.plot(x='Date', y=COM_MA+['Crude Oil'])
    plt.title("Commodities Moving Average VS Crude Oil")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0))
    plt.grid(b=True)
    plt.savefig('3_' + fig_path, bbox_inches='tight')

    plt.figure()
    tmp_df.plot(x='Date', y=TEXT+['Crude Oil'])
    plt.title("Text VS Crude Oil")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0))
    plt.grid(b=True)
    plt.savefig('4_' + fig_path, bbox_inches='tight')

    input()
    '''

    # autocorrelation_plot(df['Crude Oil_Change'][:30])
    # plt.show()
    # input()

    ###### Baseline using only historical Crude Oil Price
    X, Y = prepare_data(df, target_feature=TARGET_FEATURE, cont_features=['Crude Oil'], window=2)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

    ###### Logistic Regression
    X_train_stacked = X_train.reshape(*X_train.shape[:1], -1)
    X_test_stacked = X_test.reshape(*X_test.shape[:1], -1)
    lr_model = LogisticRegression(
        random_state=seed, solver='liblinear',
        multi_class='auto').fit(X_train_stacked, Y_train)

    Y_lr_pred = lr_model.predict(X_test_stacked)
    Y_lr_pred = [1 if i > 0.5 else 0 for i in Y_lr_pred]

    print("="*10 + "Baseline: Classification Report for LogsticRegression:" + "="*10)
    print(classification_report(Y_test, Y_lr_pred))

    print("="*10 + "Baseline: Accuracy Report for LogsticRegression:" + "="*10)
    print(accuracy_score(Y_test, Y_lr_pred))

    print()

    input()

    X, Y = prepare_data(df, cat_features=CAT_FEATURES, cont_features=CONT_FEATURES, target_feature=TARGET_FEATURE, window=win_size)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

    _class_weight = compute_class_weight('balanced', np.asarray([0, 1]), Y_train)
    class_weight = {i: weight for i, weight in enumerate(_class_weight)}

    ###### Logistic Regression
    X_train_stacked = X_train.reshape(*X_train.shape[:1], -1)
    X_test_stacked = X_test.reshape(*X_test.shape[:1], -1)
    lr_model = LogisticRegression(
        random_state=seed, solver='liblinear',
        multi_class='auto').fit(X_train_stacked, Y_train)

    Y_lr_pred = lr_model.predict(X_test_stacked)
    Y_lr_pred = [1 if i > 0.5 else 0 for i in Y_lr_pred]

    print("="*10 + "Classification Report for LogsticRegression:" + "="*10)
    print(classification_report(Y_test, Y_lr_pred))

    print("="*10 + "Accuracy Report for LogsticRegression:" + "="*10)
    print(accuracy_score(Y_test, Y_lr_pred))

    print()

    ###### Prepare Callback
    callbacks_list = []
    early_stopping = dict(monitor='val_acc', patience=3, min_delta=0.001, verbose=1)
    model_checkpoint = dict(filepath='./weights/{val_acc:.5f}_{epoch:04d}.weights.h5',
                            save_best_only=True,
                            save_weights_only=True,
                            mode='auto',
                            period=1,
                            verbose=1)

    earlystop = EarlyStopping(**early_stopping)
    callbacks_list.append(earlystop)

    checkpoint = ModelCheckpoint(**model_checkpoint)
    callbacks_list.append(checkpoint)

    ###### Vanilla Model
    tf_session = tf.Session()
    K.set_session(tf_session)
    K.get_session().run(tf.tables_initializer())

    init_op = tf.global_variables_initializer()
    tf_session.run(init_op)

    vanilla_model = build_model_vanilla(
        x_input_shape=X_train.shape[1:],
        output_shape=1
        )

    vanilla_model.fit(
        x=X_train,
        y=Y_train,
        batch_size=batch_size,
        verbose=1,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks_list,
        class_weight=class_weight
        )

    Y_vanilla_pred = vanilla_model.predict(X_test)
    Y_vanilla_pred = [1 if i > 0.5 else 0 for i in Y_vanilla_pred]

    print("="*10 + "Classification Report for Vanilla:" + "="*10)
    print(classification_report(Y_test, Y_vanilla_pred))

    print("="*10 + "Accuracy Report for Vanilla:" + "="*10)
    print(accuracy_score(Y_test, Y_vanilla_pred))
    print()

    ###### GRU Model
    tf_session = tf.Session()
    K.set_session(tf_session)
    K.get_session().run(tf.tables_initializer())

    init_op = tf.global_variables_initializer()
    tf_session.run(init_op)

    gru_model = build_model_gru(
        x_input_shape=X_train.shape[1:],
        output_shape=1
        )

    gru_model.fit(
        x=X_train,
        y=Y_train,
        batch_size=batch_size,
        verbose=1,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks_list,
        class_weight=class_weight
        )

    Y_gru_pred = gru_model.predict(X_test)
    Y_gru_pred = [1 if i > 0.5 else 0 for i in Y_gru_pred]

    print("="*10 + "Classification Report for GRU:" + "="*10)
    print(classification_report(Y_test, Y_gru_pred))

    print("="*10 + "Accuracy Report for GRU:" + "="*10)
    print(accuracy_score(Y_test, Y_gru_pred))

    print()

    ###### LSTM Model
    tf_session = tf.Session()
    K.set_session(tf_session)
    K.get_session().run(tf.tables_initializer())

    init_op = tf.global_variables_initializer()
    tf_session.run(init_op)

    lstm_model = build_model_lstm(
        x_input_shape=X_train.shape[1:],
        output_shape=1)

    lstm_model.fit(
        x=X_train,
        y=Y_train,
        batch_size=batch_size,
        verbose=1,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks_list,
        class_weight=class_weight
        )

    Y_lstm_pred = lstm_model.predict(X_test)
    Y_lstm_pred = [1 if i > 0.5 else 0 for i in Y_lstm_pred]

    print("="*10 + "Classification Report for LSTM:" + "="*10)
    print(classification_report(Y_test, Y_lstm_pred))

    print("="*10 + "Accuracy Report for LSTM:" + "="*10)
    print(accuracy_score(Y_test, Y_lstm_pred))

    print()

    ######### Random Forest
    X_train_stacked = X_train.reshape(*X_train.shape[:1], -1)
    X_test_stacked = X_test.reshape(*X_test.shape[:1], -1)

    rfc_model = RandomForestClassifier(
        n_jobs=4, 
        criterion="entropy",
        max_depth=20,
        n_estimators=100,
        max_features='sqrt',
        random_state=seed,
        )

    rfc_model.fit(X_train_stacked, Y_train)
    Y_rfc_pred = rfc_model.predict_proba(X_test_stacked)[:, 1]
    Y_rfc_pred = [1 if i > 0.5 else 0 for i in Y_rfc_pred]

    print("="*10 + "Classification Report for RandomForest:" + "="*10)
    print(classification_report(Y_test, Y_rfc_pred))

    print("="*10 + "Accuracy Report for RandomForest:" + "="*10)
    print(accuracy_score(Y_test, Y_rfc_pred))

    print()

    ###### XGBoost
    X_train_stacked = X_train.reshape(*X_train.shape[:1], -1)
    X_test_stacked = X_test.reshape(*X_test.shape[:1], -1)

    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5)
    xgb_model = xgb_model.fit(X_train_stacked, Y_train)
    Y_xgb_pred = xgb_model.predict(X_test_stacked)

    print("="*10 + "Classification Report for XGBoost:" + "="*10)
    print(classification_report(Y_test, Y_xgb_pred))

    print("="*10 + "Accuracy Report for XGBoost:" + "="*10)
    print(accuracy_score(Y_test, Y_xgb_pred))

    print()

    ###### LGBM
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=seed)
    X_train_stacked = X_train.reshape(*X_train.shape[:1], -1)
    X_val_stacked = X_val.reshape(*X_val.shape[:1], -1)
    X_test_stacked = X_test.reshape(*X_test.shape[:1], -1)

    lgb_train = lgb.Dataset(X_train_stacked, Y_train)
    lgb_val = lgb.Dataset(X_val_stacked, Y_val, reference=lgb_train)

    params = {
        'task': 'train',
        'objective': 'binary',
        'metric': 'binary_error',
        'verbose': 1}

    lgbm_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=50,
        early_stopping_rounds=10,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'eval']
        )

    Y_lgbm_pred = lgbm_model.predict(X_test_stacked)
    Y_lgbm_pred = [1 if i > 0.5 else 0 for i in Y_lgbm_pred]

    print("="*10 + "Classification Report for LGBM:" + "="*10)
    print(classification_report(Y_test, Y_lgbm_pred))

    print("="*10 + "Accuracy Report for LGBM:" + "="*10)
    print(accuracy_score(Y_test, Y_lgbm_pred))

    print()

    ######### Ensemble Learning    
    # Y_rfc_pred_train = rfc_model.predict(X_train_stacked)
    # Y_rfc_pred_train = np.asarray([1 if i > 0.5 else 0 for i in Y_rfc_pred_train])
    # Y_rfc_pred_train = Y_rfc_pred_train.reshape(Y_rfc_pred_train.shape[0], 1)
    # Y_xgb_pred_train = xgb_model.predict(X_train_stacked)
    # Y_xgb_pred_train = np.asarray([1 if i > 0.5 else 0 for i in Y_xgb_pred_train])
    # Y_xgb_pred_train = Y_xgb_pred_train.reshape(Y_xgb_pred_train.shape[0], 1)
    # Y_lgbm_pred_train = lgbm_model.predict(X_train_stacked)
    # Y_lgbm_pred_train = np.asarray([1 if i > 0.5 else 0 for i in Y_lgbm_pred_train])
    # Y_lgbm_pred_train = Y_lgbm_pred_train.reshape(Y_lgbm_pred_train.shape[0], 1)
    # X_ensemble_train = np.hstack([Y_rfc_pred_train, Y_xgb_pred_train, Y_lgbm_pred_train])

    # Y_rfc_pred = rfc_model.predict(X_test_stacked)
    # Y_rfc_pred = np.asarray([1 if i > 0.5 else 0 for i in Y_rfc_pred])
    # Y_rfc_pred = Y_rfc_pred.reshape(Y_rfc_pred.shape[0], 1)
    # Y_xgb_pred = xgb_model.predict(X_test_stacked)
    # Y_xgb_pred = np.asarray([1 if i > 0.5 else 0 for i in Y_xgb_pred])
    # Y_xgb_pred = Y_xgb_pred.reshape(Y_xgb_pred.shape[0], 1)
    # Y_lgbm_pred = lgbm_model.predict(X_test_stacked)
    # Y_lgbm_pred = np.asarray([1 if i > 0.5 else 0 for i in Y_lgbm_pred])
    # Y_lgbm_pred = Y_lgbm_pred.reshape(Y_lgbm_pred.shape[0], 1)
    # X_ensemble_test = np.hstack([Y_rfc_pred, Y_xgb_pred, Y_lgbm_pred])

    # ###### Vaniall NN
    # ensemble_model = build_model_vanilla(
    #     x_input_shape=X_ensemble_train.shape[1:],
    #     output_shape=1,
    #     flatten=False
    #     )

    # ensemble_model.fit(
    #     x=X_ensemble_train,
    #     y=Y_train,
    #     batch_size=8,
    #     verbose=1,
    #     epochs=epochs,
    #     validation_split=0.1,
    #     callbacks=callbacks_list,
    #     # class_weight=class_weight
    #     )

    # ###### Logistic Regression
    # ensemble_model = LogisticRegression(
    #     random_state=seed, solver='liblinear',
    #     multi_class='auto').fit(X_ensemble_train, Y_train)

    # ##### SVC
    # ensemble_model = SVC(gamma='scale')
    # ensemble_model.fit(X_ensemble_train, Y_train)

    # Y_pred = ensemble_model.predict(X_ensemble_test)
    # Y_pred = [1 if i > 0.5 else 0 for i in Y_pred]

    ###### Majority Voting Scheme
    # Y_pred = [1 if sum([Y_vanilla_pred[i], Y_gru_pred[i], Y_rfc_pred[i], Y_xgb_pred[i], Y_lgbm_pred[i]]) > 2 else 0 for i in range(Y_test.shape[0])]
    Y_pred = [1 if sum([
        Y_lr_pred[i],
        Y_vanilla_pred[i],
        Y_gru_pred[i],
        Y_lstm_pred[i],
        Y_rfc_pred[i],
        Y_xgb_pred[i],
        Y_lgbm_pred[i]]) > 3 else 0 for i in range(Y_test.shape[0])]

    print("="*10 + "Classification Report for Ensemble:" + "="*10)
    print(classification_report(Y_test, Y_pred))

    print("="*10 + "Accuracy Report for Ensemble:" + "="*10)
    print(accuracy_score(Y_test, Y_pred))

#end def

if __name__ == '__main__': main()

