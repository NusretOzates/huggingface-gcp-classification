import os
import subprocess
from datetime import datetime

import numpy as np
from google.cloud import aiplatform, storage
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from google.cloud.aiplatform.models import Prediction
from google.cloud.storage.bucket import Bucket
from transformers import AutoTokenizer

os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = r"C:\Users\nozat\Downloads\ageless-wall-364306-8f97a4df0351.json"

PROJECT_NAME = "ageless-wall-364306"
BUCKET_NAME = "ageless-wall-364306-vertex-ai"
REPOSITORY_NAME = "vertex-ai-images"
LOCATION = "europe-west4"


def upload_to_gcs():
    gcs_client = storage.Client(PROJECT_NAME)
    gcs_bucket: Bucket = gcs_client.bucket(BUCKET_NAME)

    train_blob = gcs_bucket.blob("tweet_eval_emotions/data/train/train.csv")
    test_blob = gcs_bucket.blob("tweet_eval_emotions/data/test/test.csv")
    validation_blob = gcs_bucket.blob(
        "tweet_eval_emotions/data/validation/validation.csv"
    )

    train_blob.upload_from_filename("train.csv")
    test_blob.upload_from_filename("test.csv")
    validation_blob.upload_from_filename("validation.csv")


# upload_to_gcs(bucket)


aiplatform.init(project=PROJECT_NAME, location=LOCATION, staging_bucket=BUCKET_NAME)

metric_spec = {"accuracy": "maximize"}

parameter_spec = {
    "lr": hpt.DoubleParameterSpec(min=0.001, max=1, scale="log"),
    "epochs": hpt.IntegerParameterSpec(min=1, max=3, scale="linear"),
}

# I'm assuming the reader created a docker artifact at the artifact registry with a name 'vertex-ai-images'
IMAGE_URI = (
    f"{LOCATION}-docker.pkg.dev/{PROJECT_NAME}/vertex-ai-images/tweet_eval:hypertune"
)

"""
RUN ONLY ONCE THIS TAKES FOREVER
Actually run this command on the gcp console "gcloud auth configure-docker europe-west4-docker.pkg.dev"
and after that run these commands below in the cloud console. Don't bother with running these locally in python code.

Or run this code in Vertex AI workbench and as a bonus not deal with credentials we set above
https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling#auth

"""

# build_result = subprocess.run(['docker', 'build', '-t', IMAGE_URI, 'custom_training_docker'])
# push_result = subprocess.run(['docker', 'push', IMAGE_URI], check=True)


# First one is the chief and the second one is workers. We have 1 chief and 2 workers but the training will be on
# both chief and workers. https://codelabs.developers.google.com/vertex_multiworker_training#5

container_spec = {
    "image_uri": IMAGE_URI,
    "args": [
        f"--project_name={PROJECT_NAME}",
        f"--bucket_name={BUCKET_NAME}",
        f"--train_path=tweet_eval_emotions/data/train/train.csv",
        f"--test_path=tweet_eval_emotions/data/test/test.csv",
        f"--validation_path=tweet_eval_emotions/data/validation/validation.csv",
        f"--distribute=multiworker",
        f"--batch_size=32",
        f"--hp=True",
    ],
}

machine_spec = {
    "machine_type": "n1-standard-4",
    # "accelerator_type": "NVIDIA_TESLA_T4",
    # "accelerator_count": 2,
}

worker_pool_specs = [
    {
        "machine_spec": machine_spec,
        "replica_count": 1,
        "container_spec": container_spec,
    },
    {
        "machine_spec": machine_spec,
        "replica_count": 2,
        "container_spec": container_spec,
    },
]

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

JOB_NAME = "custom_nlp_training-hyperparameter-job " + TIMESTAMP

custom_job = aiplatform.CustomJob(
    display_name=JOB_NAME, project=PROJECT_NAME, worker_pool_specs=worker_pool_specs
)

hp_job = aiplatform.HyperparameterTuningJob(
    display_name=JOB_NAME,
    custom_job=custom_job,
    metric_spec=metric_spec,
    parameter_spec=parameter_spec,
    max_trial_count=2,
    parallel_trial_count=2,
    project=PROJECT_NAME,
)

hp_job.run()

metrics = [trial.final_measurement.metrics[0].value for trial in hp_job.trials]
best_trial = hp_job.trials[metrics.index(max(metrics))]
best_accuracy = float(best_trial.final_measurement.metrics[0].value)
best_values = {param.parameter_id: param.value for param in best_trial.parameters}

print(best_trial)
print(best_accuracy)
print(best_values)

# The difference is not only /training /prediction. train image's name starts with tf, deploy image starts with tf2
DEPLOY_IMAGE = "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-9:latest"

print("Deployment:", DEPLOY_IMAGE)

MACHINE_TYPE = "n1-standard"
VCPU = "4"

TRAIN_COMPUTE = MACHINE_TYPE + "-" + VCPU
print("Train machine type", TRAIN_COMPUTE)

VCPU = "4"
DEPLOY_COMPUTE = MACHINE_TYPE + "-" + VCPU
print("Deploy machine type", DEPLOY_COMPUTE)

container_job = aiplatform.CustomContainerTrainingJob(
    display_name="custom_nlp_training",
    container_uri=IMAGE_URI,
    model_serving_container_image_uri=DEPLOY_IMAGE,
    project=PROJECT_NAME,
)

container_spec['args'].pop()
container_spec['args'].append(f"--hp=False")
container_spec["args"].append(f"--lr={best_values['lr']}")
container_spec["args"].append(f"--epochs={int(best_values['epochs'])}")

model = container_job.run(
    model_display_name=f"tweet_eval_{TIMESTAMP}",
    args=container_spec["args"],
    replica_count=3,
    machine_type=TRAIN_COMPUTE,
    sync=True,
)

# Create an endpoint
endpoint = model.deploy(machine_type=DEPLOY_COMPUTE, sync=True)

tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

example_text = tokenizer("I love you", truncation=True, padding="max_length")
example_text.pop("token_type_ids")

# Get prediction from the endpoint
prediction: Prediction = endpoint.predict(instances=[example_text])

print(prediction.predictions[0])
index = np.argmax(prediction.predictions[0])

id_to_label = {0: "anger", 1: "joy", 2: "optimism", 3: "saddness"}

print(id_to_label[index])
