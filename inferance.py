import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import LabelEncoder



def load_csv(file_path):
    # CSV dosyasını yükle
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Step 1: Encoding categorical columns
    categorical_cols = ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type']
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    # Step 2: Split data into X and y
    X, y = data.drop(['id','click'], axis=1), data['click']

    feat_cols = list(X.columns)
    label_name = ['click']

    # Step 3: Create feature index
    feat_idx, i = [], 0
    for ix in feat_cols:
        feat_idx.append(i)
        i += 1

    print("Feature Columns:", feat_cols,"\n",
        "feat_idx:", feat_idx,"\n",
        "Label:", label_name
        )
    
    # Step 4: Convert data to numpy arrays
    X_values, y_values = X.values, y.values

    return X_values, y_values, feat_cols, feat_idx, label_name


def parse_array_data(data):
    feat_idx = data['feat_idx']
    feat_val = data['feat_val']

    # Assuming feat_idx and X_values[i] are numpy arrays
    feat_idx = tf.cast(feat_idx, dtype=tf.int64)
    feat_val = tf.cast(feat_val, dtype=tf.float32)

    # Normalize numerical features (Same as in data_loader.py->train code)
    feat_val_signal = tf.sign(feat_val)
    norm_val = tf.math.log(tf.math.abs(feat_val) + 1.0) + 1.0
    norm_val2 = tf.maximum(2.0, norm_val)
    feat_val = tf.minimum(tf.math.abs(feat_val), norm_val2)
    feat_val = feat_val * feat_val_signal

    return feat_idx, feat_val


def predict(model, sess, data_iter_handle):
    scores = []

    while True:
        try:
            batch_scores = sess.run(
                                                    [model.scores],
                                                    feed_dict={model.handle:data_iter_handle,
                                                    model.is_train: False}
                                                    )
            scores.extend(batch_scores)
        except tf.errors.OutOfRangeError:
            break
    return scores




def main():
    # Data paths
    test_file_path = '/home/sems/Documents/GitHub/customFINT-ctr/RESULT/test.csv'

    model_dir1 = '/home/sems/Documents/GitHub/customFINT-ctr/RESULT/fint0/avazu_tmp'
    checkpoint_path1 = model_dir1 + "/model.ckpt-245000"

    model_dir2 = '/home/sems/Documents/GitHub/customFINT-ctr/RESULT/mlp0/avazu_tmp'
    checkpoint_path2 = model_dir2 + "/model.ckpt-80000"

    # Load FINT model
    graph1 = tf.Graph()
    with graph1.as_default():
        # Load the meta graph
        model1 = tf.train.import_meta_graph(checkpoint_path1 + ".meta")

        """
    # Load MLP model
    graph2 = tf.Graph()
    with graph2.as_default():
        # Load the meta graph
        model2 = tf.train.import_meta_graph(checkpoint_path2 + ".meta")

    # Create a session for the MLP model
    with tf.Session(graph=graph2) as sess2:
        # Load the variables
        model2.restore(sess2, checkpoint_path2)"""

    # Create a session for the FINT model
    with tf.Session(graph=graph1) as sess1:
        # Load the variables
        model1.restore(sess1, checkpoint_path1)

        # Load test.csv file
        data = load_csv(test_file_path)
        print(data.head())

        # Preprocess data
        X_values, y_values, feat_cols, feat_idx, label = preprocess_data(data)

        # Simulate prediction -----------------------------------------------------
        for i in range(15):

            dataset = tf.data.Dataset.from_tensor_slices({"feat_idx": feat_idx,
                                                        "feat_val": X_values[i]}
                                                        )
            
            dataset = dataset.map(parse_array_data)
            dataset = dataset.batch(batch_size=1)
            
            # Create an iterator
            data_iter = dataset.make_initializable_iterator()

            data_iter_handle1 = sess1.run(data_iter.string_handle())
            sess1.run(data_iter.initializer)
            scores1, labels1 = predict(model1, sess1, data_iter_handle1)
            """
            data_iter_handle2 = sess2.run(data_iter.string_handle())
            sess2.run(data_iter.initializer)
            scores2, labels2 = predict(model2, sess2, data_iter_handle2)"""

            print(f"-----------------------{i}-------------------------")
            print("Input: ", X_values[i])
            print("FINT Model Prediction: ","\n", scores1, "\n", labels1)
            #print("MLP Model Prediction: ", "\n", scores2, "\n", labels2)
            print("Actual Label: ", y_values[i])
            print("-------------------------------------------------")


    


main()