terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "3.87.0"
    }
  }
}

provider "google" {
  project = var.project_id
}

resource "google_project_service" "run_api" {
  service = "run.googleapis.com"
  disable_on_destroy = true
}

resource "google_project_service" "sql_api" {
  service = "sqladmin.googleapis.com"
  disable_on_destroy = true
}

resource "google_service_account" "label_studio_worker" {
  account_id   = "label-studio-worker"
  display_name = "Label Studio"
}

resource "google_project_iam_binding" "service_permissions" {
  for_each = toset([
    "logging.logWriter", "cloudsql.client"
  ])

  role       = "roles/${each.key}"
  members    = ["serviceAccount:${google_service_account.label_studio_worker.email}"]
  depends_on = [google_service_account.label_studio_worker]
}

output "service_url" {
  value = google_cloud_run_service.run_service.status[0].url
}

output "conn_str" {
  value = google_sql_database_instance.instance.connection_name
}
