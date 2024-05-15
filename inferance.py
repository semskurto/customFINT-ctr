import tensorflow as tf
import pandas as pd
from importlib import import_module


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
    broken_label = tf.constant(0, dtype=tf.float32)

    # Normalize numerical features (Same as in data_loader.py->train code)
    feat_val_signal = tf.sign(feat_val)
    norm_val = tf.math.log(tf.math.abs(feat_val) + 1.0) + 1.0
    norm_val2 = tf.maximum(2.0, norm_val)
    feat_val = tf.minimum(tf.math.abs(feat_val), norm_val2)
    feat_val = feat_val * feat_val_signal

    return broken_label, feat_idx, feat_val


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


params1 = {
    'field_num': 22,
    'lr': 0.001,
    'l2_reg': 0,
    'dropout_rate': 0,
    'batch_size': 1,
    'emb_size': 16
    }

params2 = {
    'field_num': 22,
    'lr': 0.001,
    'l2_reg': 0,
    'dropout_rate': 0.3,
    'batch_size': 1,
    'emb_size': 32
    }

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



    # Load FINT model
    # Yeni bir tf.Session oluşturun
    with tf.Session() as sess:
        # .meta dosyasından grafiği geri yükleyin
        saver = tf.train.import_meta_graph(checkpoint_path1 + '.meta')
        graph = tf.get_default_graph()

        # .index ve .data dosyalarından değişkenleri geri yükleyin (bu dosyaların adları modeliniz.meta dosyasındaki checkpoint dosyasında belirtilmiştir)
        saver.restore(sess, checkpoint_path1)

        
        # Input ve Output olabilecek tensörleri filtreleyin (op.type ile)
        for op in graph.get_operations():
            if op.type == "Placeholder" or op.type == "Placeholder_1":
                print("Input (Placeholder):", op.name)
            elif op.type in ["Sigmoid"]:  # ["Softmax", "Sigmoid", "Identity"]Tahmin katmanının türüne göre değiştirin
                print("Output:", op.name)

        
        # Grafiğinizdeki giriş ve çıktı tensörlerini (placeholder'ları) bulun
        input_tensor = tf.get_default_graph().get_tensor_by_name("Placeholder:0")  # "placeholder_ismi" yerine kendi placeholder'ınızın adını yazın
        is_train_tensor = tf.get_default_graph().get_tensor_by_name("is_train:0") # "is_train" yerine kendi is_train placeholder'ınızın adını yazın

        output_tensor = tf.get_default_graph().get_tensor_by_name("fint/Sigmoid:0")     # "output_ismi" yerine kendi output tensörünüzün adını yazın

        predictions = []
        # Simulate prediction -----------------------------------------------------
        for i in range(15):

            dataset = tf.data.Dataset.from_tensors({
                "feat_idx": tf.reshape(feat_idx, (-1, 1)),
                "feat_val": tf.reshape(X_values[i], (-1, 1))
            }).batch(1) 
            dataset = dataset.map(parse_array_data)

            data_iter = dataset.make_initializable_iterator()

            # Tahmin yap ve sonucu kaydet
            handle = sess.run(data_iter.string_handle())
            sess.run(data_iter.initializer)  # Her iterasyonda dataset'i baştan başlat
            prediction = sess.run(output_tensor, feed_dict={input_tensor: handle,
                                                            is_train_tensor: False})
            predictions.append(prediction)

    print(predictions)


    


main()