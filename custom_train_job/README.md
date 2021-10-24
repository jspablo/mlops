# Binary text classification with small BERT

### Clone project in Cloud Shell

```git clone <REPO>```

### Navigate into gcp_ai_platform_custom_container folder

```cd <REPO>/train/gcp_ai_platform_custom_container```

### Set project variables

```
export PROJECT_ID=<PROJECT_ID>
export REGION=europe-west1
export ARTIFACT_REPO=<ARTIFACT_REPO>
export BUCKET_NAME=<BUCKET_NAME>
export IMAGE_NAME=<IMAGE_NAME>
export JOB_NAME=<JOB_NAME>
```

### Set project
```gcloud config set project $PROJECT_ID```

### Create bucket in order to save log files and model
```gsutil mb -c standard -l europe-west1 gs://$BUCKET_NAME```


### Create an Artifact Regitry repo

```
gcloud artifacts repositories create $ARTIFACT_REPO --repository-format docker --location $REGION
```

### Create a docker image with the code
```
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$IMAGE_NAME --timeout 30m
```

### Create a custom Vertex AI training job 
```
gcloud beta ai custom-jobs create --region=$REGION --display-name=$JOB_NAME --worker-pool-spec=machine-type=n1-highmem-2,accelerator-type=NVIDIA_TESLA_K80,replica-count=1,container-image-uri=$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$IMAGE_NAME --args=output_dir=$BUCKET_NAME
```
