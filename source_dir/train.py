
import argparse
import os

import tensorflow as tf

import subprocess
import sys

# Installing tf add-ons to obtain Confusion Matrix in the callback metrics 
def install(package):
    subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package])
    
install('tensorflow-addons')
print("Installed tf add-ons")

import tensorflow_addons as tfa

from pathlib import Path

import json
import numpy as np
from tensorflow import keras

import pickle


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Device:", tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print("Number of replicas:", strategy.num_replicas_in_sync)

CLASSES = [
    # 'notvisible', filtered out
    'dacia',
    'renault',
    'peugeot',
    'fiat',
    'hyundai',
    'volkswagen',
    'citroen',
    'ford',
    'other' # encompass all other brands that have less than 50 training samples
]

class ConfusionMatrixCallBack(keras.callbacks.Callback):
    def __init__(self, class_id2label):
        super(ConfusionMatrixCallBack, self).__init__()
        self.class_id2label = class_id2label
        
    def on_epoch_end(self, epoch, logs=None):
        print("Epoch {} of training".format(epoch))
        print(f"##### Confusion matrix ########")
        if logs:
            for class_id, conf_matrix in enumerate(logs['multiConfusion']):
                print(f"Class {class_id} - {self.class_id2label[class_id]}:")
                print(f"{conf_matrix}")

            
    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print(f"Available log keys: {keys}")
        print(f"##### Confusion matrix ########")
        for class_id, conf_matrix in enumerate(logs['multiConfusion']):
            print(f"Class {class_id} - {self.class_id2label[class_id]}:")
            print(f"{conf_matrix}")
            
        
def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    return image


def read_tfrecord(example):
    tfrecord_format = (
        {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/class/brand': tf.io.FixedLenFeature([], tf.string),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image/encoded'])
    label = tf.cast(example['image/class/brand'], tf.string)
    return image, label

def pre_process(image, label):
    
    # label pre-processing: string -> one-hot vector
    output_label = string2classid.lookup(label)
    output_label = tf.one_hot(
      indices=output_label, depth=len(CLASSES), on_value=1.0, off_value=0.0)
    return image, output_label   

def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        read_tfrecord, num_parallel_calls=AUTOTUNE
    )
    return dataset

def filter_dataset(image, label):
    # Remove samples with label equals to 'notvisible'
    return label != "notvisible"

def get_dataset(filenames, shuffle=True):
    dataset = load_dataset(filenames)
    
    # Filter
    dataset = dataset.filter(filter_dataset)
    
    # Preprocessing
    dataset = dataset.map(
      pre_process, num_parallel_calls=AUTOTUNE
    )
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

def make_overfit_model():
    """Creates a model without regularization, data augmentation.
    Used to test if learning is happening on a base model before iterating.

    """
    base_model = tf.keras.applications.Xception(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )

    base_model.trainable = False

    inputs = tf.keras.layers.Input([*IMAGE_SIZE, 3])
    
    # Preprocess_input steps:
    # tf32
    # cast to -1, 1
    x = tf.keras.applications.xception.preprocess_input(inputs)

    x = base_model(x)
        
    # -------------- New head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(
                name='acc', dtype=None
            ),
            tfa.metrics.MultiLabelConfusionMatrix(
                num_classes=len(CLASSES),
                name='multiConfusion', dtype=None
            )
        ],
    )

    return model

def make_model():
    base_model = tf.keras.applications.Xception(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    # ------------ data augmentation
    data_augmentation = keras.Sequential(
       [
           tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
           tf.keras.layers.experimental.preprocessing.RandomRotation(0.25),
           tf.keras.layers.experimental.preprocessing.RandomContrast(.2)
       ]
    )

    base_model.trainable = False

    inputs = tf.keras.layers.Input([*IMAGE_SIZE, 3])
    
    x = data_augmentation(inputs)
    
    # Preprocessing here already includes normalization between -1 and 1
    x = tf.keras.applications.xception.preprocess_input(x)
    x = base_model(x)
    
    # TODO
    # 1. overfit [OK]
    #  run on only one batch (overfit keras option) + obtain the confusion matrix 
    #   freeze and unfreeze
    
    # 2. Remove regularization [ONGOING]
    #   - dropout
    #   - data augmentation
    # 3. Regularize
    
    # 4. Error Analysis
    #  examine 100 images with predictions
        
    # -------------- New head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(
                name='acc', dtype=None
            ),
            tfa.metrics.MultiLabelConfusionMatrix(
                num_classes=len(CLASSES),
                name='multiConfusion', dtype=None
            )
        ],
    )

    return model


if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    #============= CLI Commands =================
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--image_size', type=int, default=600)
    parser.add_argument('--overfit', type=bool, default=False)
    parser.add_argument('--small_data', type=bool, default=False)

    # input data and model directories
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args, _ = parser.parse_known_args()
    
    model_artefacts_output_path = os.environ['SM_MODEL_DIR']
    other_artefacts_output_path = os.environ['SM_OUTPUT_DATA_DIR']
    
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", args.train)
    
    #============= Hyperparams =================
    AUTOTUNE = tf.data.AUTOTUNE
    BATCH_SIZE = args.batch_size
    IMAGE_SIZE = [int(args.image_size), int(args.image_size)]

    label_to_class_id = {
        label: class_id for class_id, label in enumerate(sorted(CLASSES))
    }

    print("Writing mapping label to class id")
    with open(os.path.join(other_artefacts_output_path, "label_2_class_id.txt"), "w") as f:
        json.dump(label_to_class_id, f)

    initializer = tf.lookup.KeyValueTensorInitializer(
        keys=CLASSES,
        values=tf.cast(tf.range(len(CLASSES)), tf.int32),
        key_dtype=tf.string,
        value_dtype=tf.int32
    )
    # Remark: if key is not found (class not considered in the classifcation task), 
    # assign the label len(CLASSES)-1 corresponding to 'other' class.
    string2classid = tf.lookup.StaticHashTable(initializer, default_value=len(CLASSES)-1)

    #============= Datasets =================
    if not args.small_data:
        train_dataset = get_dataset(
            [str(i) for i in Path(args.train).glob("*.record")]
        )
    
        val_dataset = get_dataset(
            [str(i) for i in Path(args.test).glob("*.record")],
            shuffle=False
        )
    else:
        train_dataset = get_dataset(
            [str(i) for i in Path(args.train).glob("*.record")]
        ).take(10)
    
        val_dataset = get_dataset(
            [str(i) for i in Path(args.test).glob("*.record")],
            shuffle=False
        ).take(5)

    # TODO: does not work, returns -2
    print(f"Number of training samples: {tf.data.experimental.cardinality(train_dataset)}")
    print(f"Number of test samples    : {tf.data.experimental.cardinality(val_dataset)}")

    #============= Learning rate =================
    initial_learning_rate = args.learning_rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=200, decay_rate=0.96, staircase=True
    )
    #============= Callbacks =================
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        "classifier_car_brand_{epoch}.h5", save_best_only=True
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )
    
    confusion_matrix_cb = ConfusionMatrixCallBack(
        class_id2label={v: k for k, v in label_to_class_id.items()}
    )
    
    # Create folder containing tensorboard logs, under the path that SageMaker 
    # collects and save to S3 at the end of the training
    Path(os.path.join(other_artefacts_output_path,"./logs")).mkdir(parents=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(other_artefacts_output_path,"./logs")
    )

    #============= Model =================
    with strategy.scope():
        if args.overfit:
            model = make_overfit_model()
        else:
            model = make_model()
 
    print(model.summary())

    #============= Fitting =================
    print("Starting Training")
    history = model.fit(
        train_dataset,
        # class_weight=WEIGHTS_RATIO,
        epochs=args.epochs,
        validation_data=val_dataset,
        callbacks=[checkpoint_cb, early_stopping_cb, confusion_matrix_cb, tensorboard_callback],
        verbose=2
    )
    print("Saving history")
    with open(os.path.join(other_artefacts_output_path, 'train_history_dict.pkl'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    print("Saving model")
    model.save(os.path.join(model_artefacts_output_path, "final_model"))
