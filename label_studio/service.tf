resource "google_cloud_run_service" "run_service" {
  name = "labelstudio"
  location = var.region

  template {
    spec {
      service_account_name = google_service_account.label_studio_worker.email
      container_concurrency = var.service_concurrency
      timeout_seconds = var.service_timeout
      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.service_artifact_repo}/${var.service_image}"
        resources {
          limits = {
            cpu    = "1000m"
            memory = "512Mi"
          }
        }
        env {
          name  = "DJANGO_DB"
          value = "default"
        }
        env {
          name  = "POSTGRE_USER"
          value = "postgres"
        }
        env {
          name  = "POSTGRE_PASSWORD"
          value = google_sql_user.users.password
        }
        env {
          name  = "POSTGRE_NAME"
          value = "postgres"
        }
        env {
          name  = "POSTGRE_HOST"
          value = "/cloudsql/${google_sql_database_instance.instance.connection_name}"
        }
        env {
          name  = "POSTGRE_PORT"
          value = "5432"
        }
        env {
          name = "LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK"
          value = true
        }
        env {
          name = "LABEL_STUDIO_USERNAME"
          value = var.label_studio_username
        }
        env {
          name = "LABEL_STUDIO_PASSWORD"
          value = var.label_studio_password
        }
      }
    }
    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale" = var.service_num_containers_autoscale
        "run.googleapis.com/cloudsql-instances" = google_sql_database_instance.instance.connection_name
        "run.googleapis.com/client-name" = "terraform"
      }
    }
  }

  traffic {
    percent = 100
    latest_revision = true
  }

  depends_on = [
    google_project_service.run_api,
    google_project_service.sql_api,
    google_sql_database_instance.instance,
    google_service_account.label_studio_worker
  ]
}

data "google_iam_policy" "noauth" {
  binding {
    role = "roles/run.invoker"
    members = [
      "allUsers",
    ]
  }
}

resource "google_cloud_run_service_iam_policy" "noauth" {
  location = google_cloud_run_service.run_service.location
  project  = google_cloud_run_service.run_service.project
  service  = google_cloud_run_service.run_service.name

  policy_data = data.google_iam_policy.noauth.policy_data
  depends_on  = [google_cloud_run_service.run_service]
}