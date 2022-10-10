
FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-9
WORKDIR /

# Installs hypertune library
RUN pip install  transformers datasets google-cloud-storage cloudml-hypertune

# Copies the trainer code to the docker image.
COPY trainer /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]