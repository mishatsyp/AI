import pandas as pd
from catboost import CatBoostClassifier, Pool, metrics, cv
import numpy as np
from sklearn.model_selection import train_test_split
print('start')

path = 'C:/Users/yarom/PycharmProjects/pythonNetWork/Data/Allcups_F/train_share.tsv'
train = pd.read_table(path)

path = 'C:/Users/yarom/PycharmProjects/pythonNetWork/Data/Allcups_F/test_share.tsv'
test = pd.read_table(path)
print('Data dowloaded')
categorical = ["feature_1", "feature_3", "feature_5", "feature_6", "feature_8", "feature_9", "feature_11", "feature_12", "feature_13", "feature_14", "feature_16", "feature_18", "feature_23", "feature_24", "feature_26", "feature_28", "application_2"]

def data_preprocessing(data):
    for column in categorical:
        if column != "target":
            data[column] = data[column].fillna(-1)
            data[column] = data[column].astype('str')

    numeric = list(set(train.columns) - set(categorical))
    for name in numeric:
        if name != "target":
            data[name] = data[name].fillna(0.0)

    return data

test = data_preprocessing(test).drop(columns = 'Unnamed: 0')

train = data_preprocessing(train)

data_train, data_val  = train_test_split(train, test_size=0.2, random_state=42, shuffle=True)

y_train = data_train['target']
X_train = data_train.drop(columns = 'target')
y_val = data_val['target']
X_val = data_val.drop(columns = 'target')

print('Ready training')
# num_of_test_data = int(97211*0.2)

# data_train_1, data_val_1  = train_test_split(train[train['target'] == 1], test_size=num_of_test_data, random_state=42, shuffle=True)
# data_train_0, data_val_0  = train_test_split(train[train['target'] == 0], test_size=num_of_test_data, random_state=42, shuffle=True)
#
# data_train = pd.concat([data_train_0, data_train_1])
# data_val = pd.concat([data_val_0, data_val_1])
#
# y_train = data_train.copy()['target']
# X_train = data_train.copy().drop(columns = 'target')
# y_val = data_val.copy()['target']
# X_val = data_val.copy().drop(columns = 'target')

class RecallatK:
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, targets, weight):
        threshold = int(len(targets) * 0.15)

        top_k_preds = np.argsort(approxes[0])[-threshold:]

        total_positives = np.sum(targets == 1)

        error = np.sum(targets[top_k_preds]) / total_positives

        output = (error, np.sum(weight))

        return output


model = CatBoostClassifier(
    #scale_pos_weight=scale_pos_weight_value,
    auto_class_weights='SqrtBalanced',
    #auto_class_weights='Balanced',
    l2_leaf_reg=int(4.0),
    learning_rate=0.1,
    #min_data_in_leaf = int(60.0),
    depth = int(10.0),
    loss_function='Logloss',
    eval_metric=RecallatK(),
    random_seed=42,
    logging_level='Silent',
    iterations=1000,
    task_type="GPU",
    devices='0',
    use_best_model=True
)

#model = CatBoostClassifier(learning_rate=0.001)
#base_model = model.load_model('C:/Users/yarom/PycharmProjects/pythonNetWork/NetWorks/Catboost_models/model.cbm')
#model.set_params(learning_rate=0.001)

model.fit(
    X_train, y_train,
    cat_features=categorical,
    eval_set=(X_val, y_val),
    logging_level='Verbose', # Verbose если нужен вывод обучения # Silent если не нужен вывод обучения
   # init_model='C:/Users/yarom/PycharmProjects/pythonNetWork/NetWorks/Catboost_models/model.cbm',
)


model.save_model('C:/Users/yarom/PycharmProjects/pythonNetWork/NetWorks/Catboost_models/model_post_1.cbm')

print('Predicting')
predict = model.predict_proba(test)[:,1:].flatten()
submission = pd.DataFrame({'id' : test.index, 'target' : predict})
submission.to_csv('C:/Users/yarom/PycharmProjects/pythonNetWork/Data/Allcups_F/submission2.csv', index=False)