variable "project_id" {
    type = string
    description = "Google Cloud Platform Project ID"
}

variable "region" {
    type = string
    default = "europe-west1"
}

variable "db_name" {
    type = string
    default = "sql-labelstudio"
}
variable "db_user" {
    type = string
    default = "postgres"
}

variable "db_password" {
    type = string
}

variable "service_artifact_repo" {
    type = string
}

variable "service_image" {
    type = string
}

variable "service_concurrency" {
    type = number
    default = 8
}

variable "service_timeout" {
    type = number
    default = 300
}

variable "service_num_containers_autoscale" {
    type = number
    default = 1
}

variable "label_studio_username" {
    type = string
    default = "testuser@gmail.com"
}

variable "label_studio_password" {
    type = string
}
