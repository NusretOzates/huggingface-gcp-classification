# Tuning and Deploying HF Transformers with Vertex AI

This repository contains code to train, tune and deploy a Hugging Face Transformer model with Vertex AI.

`custom_training_docker` - Dockerfile and training code for custom training container

`training.py` - Creation of the custom training container, hyperparameter-tuning job ,training job and model deployment

Make sure you run "gcloud auth application-default login" before running the training.py file. or you can use the service account key json file.

I've created a medium article to explain the code in this repository. You can find it here.

Part 1: https://medium.com/@m.nusret.ozates/tuning-and-deploying-hf-transformers-with-vertex-ai-part-1-preparing-prerequisites-and-dataset-9794ebe8e291

Part 2: https://medium.com/@m.nusret.ozates/tuning-and-deploying-hf-transformers-with-vertex-ai-part-2-training-code-591186445a2a

Part 3: https://medium.com/@m.nusret.ozates/tuning-and-deploying-huggingface-transformers-with-vertex-ai-part-3-start-distributed-tuning-and-e0943bfc9d4b

