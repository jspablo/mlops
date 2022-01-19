# mlops

A set of common machine learning tasks solved using ml ops practices.

**custom_train_job**: trains a model by packing dependencies and testing code locally using docker. Pushes container to cloud, as a Vertex AI custom job, to take advantage of GPUs.

**hf_pipeline**: a model training pipeline using Kubeflow and Vertex AI pipelines. Registers model metrics and parameters between executions for experiment tracking.

**label_studio**: deploys a labeling tool with Terraform [based on this article](https://dev.to/marjoripomarole/bootstrapping-label-studio-on-google-cloud-with-terraform-43ja). Serves the GUI as a serverless app using Cloud Run. Annotations are saved using Cloud SQL and Cloud Storage. It exposes new variables for app scaling and security.
