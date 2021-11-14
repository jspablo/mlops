resource "google_sql_user" "users" {
  name     = var.db_user
  instance = google_sql_database_instance.instance.name
  password = var.db_password
  deletion_policy = "ABANDON"
}

resource "google_sql_database_instance" "instance" {
  name   = var.db_name
  database_version = "POSTGRES_13"
  region = var.region
  settings {
    tier = "db-f1-micro"
  }
  deletion_protection = "false"
}