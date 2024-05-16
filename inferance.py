import tensorflow as tf
import pandas as pd
from importlib import import_module
import time


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
    broken_label = tf.constant(0.5, dtype=tf.float32)

    # Normalize numerical features (Same as in data_loader.py->train code)
    feat_val_signal = tf.sign(feat_val)
    norm_val = tf.math.log(tf.math.abs(feat_val) + 1.0) + 1.0
    norm_val2 = tf.maximum(2.0, norm_val)
    feat_val = tf.minimum(tf.math.abs(feat_val), norm_val2)
    feat_val = feat_val * feat_val_signal

    return broken_label, feat_idx, feat_val


def predict(checkpoint_path, X_values, y_values, feat_idx, model_name="fint"):
    print("Model Name:", model_name, "--------------")
    # Create a session
    with tf.Session() as sess:
        # Load the graph model
        saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
        graph = tf.get_default_graph()

        # Load the weights
        saver.restore(sess, checkpoint_path)

        
        # Find model inputs and outputs
        for op in graph.get_operations():
            if op.type == "Placeholder" or op.type == "Placeholder_1":
                print("Input (Placeholder):", op.name)
            elif op.type in ["Sigmoid"]:  # ["Softmax", "Sigmoid", "Identity"]
                print("Output:", op.name)

        
        # Set input and output tensors for prediction
        input_tensor = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        is_train_tensor = tf.get_default_graph().get_tensor_by_name("is_train:0")

        output_tensor = tf.get_default_graph().get_tensor_by_name(f"{model_name}/Sigmoid:0")

        predictions = []
        # Simulate prediction -----------------------------------------------------
        for i in range(len(X_values)-1):
            start_time = time.time()

            dataset = tf.data.Dataset.from_tensors({
                "feat_idx": feat_idx,
                "feat_val": X_values[i]
            }).batch(1) 
            dataset = dataset.map(parse_array_data)

            data_iter = dataset.make_initializable_iterator()

            handle = sess.run(data_iter.string_handle())
            sess.run(data_iter.initializer)
            prediction = sess.run(output_tensor, feed_dict={input_tensor: handle,
                                                            is_train_tensor: False})
            predictions.append(prediction)
            end_time = time.time()

            print("[1] Predicted:", prediction, "--> y_true:", y_values[i])
            print("[2] Inferance Time:", (end_time - start_time) * 1000, "ms")

    tf.reset_default_graph()
    print("-----------------*-----------------")


def main():
    # Data paths
    test_file_path = '/home/sems/Documents/GitHub/customFINT-ctr/RESULT/test.csv'

    # Load test.csv file
    data = load_csv(test_file_path)
    print(data.head())

    # Preprocess data
    X_values, y_values, feat_cols, feat_idx, label = preprocess_data(data)

    model_dir1 = '/home/sems/Documents/GitHub/customFINT-ctr/RESULT/fint0/avazu_tmp'
    checkpoint_path1 = model_dir1 + '/model.ckpt-245000'

    model_dir2 = '/home/sems/Documents/GitHub/customFINT-ctr/RESULT/mlp0/avazu_tmp'
    checkpoint_path2 = model_dir2 + '/model.ckpt-80000'



    # Load FINT model and predict
    predict(checkpoint_path1, X_values, y_values, feat_idx)

    # Load MLP model and predict
    predict(checkpoint_path2, X_values, y_values, feat_idx, model_name="mlp")



main()