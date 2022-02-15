resource "digitalocean_droplet" "fastapitorchraw" {
    image = var.vm_image
    name = "fastapitorchraw"
    region = "lon1"
    size = "s-4vcpu-8gb-intel"
    ssh_keys = [
      data.digitalocean_ssh_key.id_rsa.id
    ]

    connection {
        host = self.ipv4_address
        user = "root"
        type = "ssh"
        private_key = file(var.pvt_key)
        timeout = "2m"
    }

    provisioner "remote-exec" {
        inline = [
            "export PATH=$PATH:/usr/bin",
            "sudo apt update",
            "sudo apt install apt-transport-https ca-certificates curl software-properties-common",
            "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -",
            "sudo add-apt-repository \"deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable\"",
            "apt-cache policy docker-ce",
            "sudo apt install docker-ce -y",
            "sudo docker pull jspablo/fastapi_torch_raw:latest",
            "sudo docker run -d -p 80:80 jspablo/fastapi_torch_raw:latest"
        ]
    }
}

resource "digitalocean_firewall" "web" {
    name = "fastapitorchraw-80"

    droplet_ids = [digitalocean_droplet.fastapitorchraw.id]

    inbound_rule {
        protocol         = "tcp"
        port_range       = "22"
        source_addresses = ["192.168.1.0/24", "2002:1:2::/48"]
    }

    inbound_rule {
        protocol         = "tcp"
        port_range       = "80"
        source_addresses = ["0.0.0.0/0", "::/0"]
    }

    inbound_rule {
        protocol         = "tcp"
        port_range       = "443"
        source_addresses = ["0.0.0.0/0", "::/0"]
    }

    inbound_rule {
        protocol         = "icmp"
        source_addresses = ["0.0.0.0/0", "::/0"]
    }

    outbound_rule {
        protocol              = "tcp"
        port_range            = "53"
        destination_addresses = ["0.0.0.0/0", "::/0"]
    }

    outbound_rule {
        protocol              = "udp"
        port_range            = "53"
        destination_addresses = ["0.0.0.0/0", "::/0"]
    }

    outbound_rule {
        protocol              = "icmp"
        destination_addresses = ["0.0.0.0/0", "::/0"]
    }
}