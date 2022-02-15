output "fastapionnxraw_droplet" {
  value = [
    "${digitalocean_droplet.fastapionnxraw.*.ipv4_address}"
  ]
}

output "fastapitorchraw_droplet" {
  value = [
    "${digitalocean_droplet.fastapitorchraw.*.ipv4_address}"
  ]
}