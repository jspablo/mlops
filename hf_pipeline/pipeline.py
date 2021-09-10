import argparse
import os
from datetime import datetime

import kfp.gcp as gcp
from dotenv import load_dotenv
from google.cloud.aiplatform import pipeline_jobs
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component

from components.preprocess.raw.src.preprocess_raw import preprocess_raw
from components.train.src.train import train


@dsl.pipeline(
    name="hf-pipeline",
    description="Training HF token classification model",
)
def ner_pipeline(
    dataset_url: str,
    dataset_filename: str,
    pretrained_model: str,
    root_path: str,
    accelerator: str,
    batch_size: int,
    epochs: int
):
    preprocess_raw_op = component(
        func=preprocess_raw,
        base_image="python:3.7",
        packages_to_install=[
            "scikit-learn==0.24.2",
            "transformers==4.9.1"
        ]
    )

    train_op = component(
        func=train,
        base_image="tensorflow/tensorflow:2.5.1-gpu",
        packages_to_install=[
            "scikit-learn==0.24.2",
            "transformers==4.9.1"
        ]
    )

    preprocess_raw_step = preprocess_raw_op(
        root_path=root_path,
        dataset_url=dataset_url,
        dataset_filename=dataset_filename,
        pretrained_model=pretrained_model
    )

    train_step = (train_op(
        root_path=root_path,
        model=pretrained_model,
        batch_size=batch_size,
        epochs=epochs,
        unique_tags=preprocess_raw_step.outputs["unique_tags_path"],
        train_encodings=preprocess_raw_step.outputs["train_encodings_path"],
        train_labels=preprocess_raw_step.outputs["train_labels_path"],
        val_encodings=preprocess_raw_step.outputs["val_encodings_path"],
        val_labels=preprocess_raw_step.outputs["val_labels_path"],
    ).
    add_node_selector_constraint(
        'cloud.google.com/gke-accelerator', 'nvidia-tesla-t4'
    )
    .set_gpu_limit(1)
    .apply(gcp.use_preemptible_nodepool(hard_constraint=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=str)
    parser.add_argument("--package", type=str, default="hf_pipeline.json")
    parser.add_argument("--name", type=str, default="hf-pipeline")
    parser.add_argument("--accelerator", type=str, default="nvidia-tesla-t4")
    parser.add_argument("--use_env_vars", action="store_false")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--location", type=str, default="europe-west4")
    parser.add_argument("--enable_cache", action="store_false")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="distilbert-base-cased"
    )
    parser.add_argument(
        "--dataset_url",
        type=str,
        default="http://noisy-text.github.io/2017/files/wnut17train.conll"
    )
    parser.add_argument(
        "--dataset_filename",
        type=str,
        default="wnut17train.conll"
    )
    parser.add_argument(
        "--pipeline_root",
        type=str,
        default="gs://lesgopipelines/test"
    )
    args = parser.parse_args()

    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

    package_path = args.package
    project_id = args.project_id
    use_env_vars = args.use_env_vars
    accelerator = args.accelerator
    batch_size = args.batch_size
    epochs = args.epochs
    location = args.location
    display_name = args.name
    pretrained_model = args.pretrained_model
    dataset_filename = args.dataset_filename
    dataset_url = args.dataset_url
    job_id = f"{display_name}-{TIMESTAMP}"
    pipeline_root_path = os.path.join(args.pipeline_root, job_id)

    compiler.Compiler().compile(
        pipeline_func=ner_pipeline,
        package_path=package_path
    )

    if use_env_vars:
        load_dotenv()

    if project_id and location:
        run = pipeline_jobs.PipelineJob(
            project=project_id,
            location=location,
            display_name=display_name,
            template_path=package_path,
            job_id=job_id,
            pipeline_root=pipeline_root_path,
            parameter_values={
                "accelerator": accelerator,
                "batch_size": batch_size,
                "epochs": epochs,
                "dataset_filename": dataset_filename, 
                "dataset_url": dataset_url,
                "pretrained_model": pretrained_model,
                "root_path": os.path.join(
                    "/gcs",
                    pipeline_root_path.split("://")[1]
                )
            },
            enable_caching=True
        )

        pipeline_job = run.run(sync=False)
