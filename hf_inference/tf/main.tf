terraform {
    required_providers {
        digitalocean = {
        source = "digitalocean/digitalocean"
        version = "~> 2.0"
        }
    }
}

variable "do_token" {
    description = "Digital Ocean API Token"
}

variable "pvt_key" {
    description = "Private SSH key location"
    default = "~/.ssh/id_rsa"
}

variable "vm_image" {
    default = "ubuntu-20-04-x64"
}

data "digitalocean_ssh_key" "id_rsa" {
    name = "id_rsa"
}

provider "digitalocean" {
    token = var.do_token
}
