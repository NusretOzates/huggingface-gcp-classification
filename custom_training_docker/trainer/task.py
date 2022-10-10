# Single, Mirrored and MultiWorker Distributed Training
import argparse
import json
import os

import datasets
import hypertune
import keras
import tensorflow as tf
from datasets import load_dataset
from google.cloud import storage
from google.cloud.storage import Bucket
from keras import Model
from keras.layers import Dense
from keras.optimizers import Adam
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TFAutoModel,
    TFPreTrainedModel,
)
from transformers.modeling_tf_outputs import TFBaseModelOutput

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-dir",
    dest="model_dir",
    default=os.getenv("AIP_MODEL_DIR"),
    type=str,
    help="Model dir.",
)
parser.add_argument("--lr", dest="lr", default=0.001, type=float, help="Learning rate.")

parser.add_argument(
    "--epochs", dest="epochs", default=5, type=int, help="Number of epochs."
)
parser.add_argument(
    "--batch_size", dest="batch_size", default=16, type=int, help="Size of a batch."
)
parser.add_argument(
    "--distribute",
    dest="distribute",
    type=str,
    default="single",
    help="distributed training strategy",
)

parser.add_argument(
    "--project_name",
    dest="project",
    type=str,
    default=os.getenv("CLOUD_ML_PROJECT_ID"),
    help="name of the project",
)
parser.add_argument(
    "--bucket_name", dest="bucket", type=str, help="name of the project"
)

parser.add_argument(
    "--train_path", dest="train", type=str, help="GCS path of the train data"
)
parser.add_argument(
    "--test_path", dest="test", type=str, help="GCS path of the test data"
)
parser.add_argument(
    "--validation_path",
    dest="validation",
    type=str,
    help="GCS path of the validation data",
)

parser.add_argument("--hp", dest="hp", type=bool, help="Are we tuning hyperparameters?")

args = parser.parse_args()

# Single Machine, single compute device
if args.distribute == "single":
    if tf.test.is_gpu_available():
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

# Single Machine, multiple compute device
elif args.distribute == "mirrored":
    strategy = tf.distribute.MirroredStrategy()

# Multi Machine, multiple compute device
elif args.distribute == "multiworker":
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
else:
    raise ValueError("Unknown distribution strategy")

tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")


def _is_chief(task_type, task_id):
    """Check for primary if multiworker training"""

    tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
    cluster = tf_config["cluster"]

    if ("chief" in cluster) and "worker" in cluster:
        return task_type == "chief"

    return (
        (task_type == "chief")
        or (task_type == "worker" and task_id == 0)
        or task_type is None
    )


def hf_to_tf(dataset: datasets.Dataset, shuffle: bool) -> tf.data.Dataset:
    """Converts HuggingFace Dataset object into a TF Dataset.

    Args:
        dataset:  HuggingFace Dataset object
        shuffle:  Whether to shuffle the dataset

    Returns:
        TF Dataset object
    """

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf", padding=False)

    NUM_WORKERS = strategy.num_replicas_in_sync
    # Here the batch size scales up by number of workers since
    # `tf.data.Dataset.batch` expects the global batch size.
    GLOBAL_BATCH_SIZE = args.batch_size * NUM_WORKERS

    return dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["labels"],
        batch_size=GLOBAL_BATCH_SIZE,
        collate_fn=data_collator,
        drop_remainder=True,
        shuffle=shuffle,
    )


def download_from_gcs():
    gcs_client = storage.Client(args.project)
    bucket: Bucket = gcs_client.bucket(args.bucket)
    train_blob = bucket.blob(args.train)
    test_blob = bucket.blob(args.test)
    validation_blob = bucket.blob(args.validation)

    train_blob.download_to_filename("train.csv")
    test_blob.download_to_filename("test.csv")
    validation_blob.download_to_filename("validation.csv")


def get_data():
    download_from_gcs()

    dataset = load_dataset(
        "csv",
        data_files={
            "train": "train.csv",
            "test": "test.csv",
            "validation": "validation.csv",
        },
    )
    dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)

    dataset = dataset.map(
        function=lambda examples: tokenizer(
            examples["text"], truncation=True, padding="max_length"
        ),
        batched=True,
    )

    tf_train = hf_to_tf(dataset["train"], True)
    tf_val = hf_to_tf(dataset["validation"], False)
    tf_test = hf_to_tf(dataset["test"], False)

    if not args.hp:
        tf_train = tf_train.concatenate(tf_val)
        tf_val = tf_test

    return tf_train, tf_val


def get_model():
    input_ids = keras.Input(
        name="input_ids",
        shape=tokenizer.init_kwargs["model_max_length"],
        dtype="int32",
    )
    attention_mask = keras.Input(
        name="attention_mask",
        shape=tokenizer.init_kwargs["model_max_length"],
        dtype="int32",
    )

    base_model: TFPreTrainedModel = TFAutoModel.from_pretrained(
        "google/electra-small-discriminator"
    )
    base_model.trainable = False

    base_model_output: TFBaseModelOutput = base_model(
        {"input_ids": input_ids, "attention_mask": attention_mask}
    )

    last_hidden_state = base_model_output.last_hidden_state
    x = keras.layers.GlobalAveragePooling1D()(last_hidden_state)

    classification_layer = Dense(4, "softmax")(x)

    model = Model(inputs=[input_ids, attention_mask], outputs=[classification_layer])
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train(model: keras.Model, train: tf.data.Dataset, validation: tf.data.Dataset):
    resolver = strategy.cluster_resolver
    task_type, task_id = resolver.task_type, resolver.task_id if resolver else (
        None,
        None,
    )

    base_callback_folder = os.getenv("AIP_CHECKPOINT_DIR")
    filepath = (
        "model-chef" if _is_chief(task_type, task_id) else f"workertemp_{task_id}"
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{base_callback_folder}{filepath}",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )

    history = model.fit(
        train,
        epochs=args.epochs,
        validation_data=validation,
        callbacks=[model_checkpoint_callback],
    )

    hp_metric = history.history["val_accuracy"][-1]

    # single, mirrored or primary for multiworker
    if _is_chief(task_type, task_id):

        if args.hp:
            hpt = hypertune.HyperTune()
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag="accuracy",
                metric_value=hp_metric,
                global_step=args.epochs,
            )

        model.save(args.model_dir)
    # non-primary workers for multi-workers
    else:
        # each worker saves their model instance to a unique temp location
        model_save_dir = args.model_dir[:-1] + "workertemp_" + str(task_id)
        tf.io.gfile.makedirs(model_save_dir)
        model.save(model_save_dir)


with strategy.scope():
    #  Model building/compiling need to be within
    # `strategy.scope()`.
    model = get_model()

train_data, validation_data = get_data()
train(model, train_data, validation_data)
