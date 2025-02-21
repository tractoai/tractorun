target "jammy_python_sys" {
  platforms = ["linux/amd64"]
  dockerfile-inline = <<EOT
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    python3 -m pip install --upgrade pip
EOT
}

target "generic_tests" {
  platforms = ["linux/amd64"]
  contexts = {
    base_image = "target:jammy_python_sys"
  }
  context           = "${PROJECT_ROOT}"
  tags = [
    "${DOCKER_REPO}/tractorun-generic-tests:${DOCKER_TAG}"
  ]
  dockerfile-inline = <<EOT
FROM base_image
COPY requirements.txt requirements_tests.txt /tmp
RUN python3 -m pip install \
  -r /tmp/requirements.txt \
  -r /tmp/requirements_tests.txt
EOT
}

target "tractorch_tests" {
  platforms = ["linux/amd64"]
  contexts = {
    base_image = "target:jammy_python_sys"
  }
  context           = "${PROJECT_ROOT}"
  tags = [
    "${DOCKER_REPO}/tractorun-tractorch-tests:${DOCKER_TAG}"
  ]
  dockerfile-inline = <<EOT
FROM base_image
COPY requirements.txt requirements_tests.txt requirements_torch.txt /tmp
RUN python3 -m pip install \
  -r /tmp/requirements.txt \
  -r /tmp/requirements_tests.txt
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cpu -r "/tmp/requirements_torch.txt"
EOT
}

target "tractorax_tests" {
  platforms = ["linux/amd64"]
  contexts = {
    base_image = "target:jammy_python_sys"
  }
  context           = "${PROJECT_ROOT}"
  tags = [
    "${DOCKER_REPO}/tractorun-tractorax-tests:${DOCKER_TAG}"
  ]
  dockerfile-inline = <<EOT
FROM base_image
COPY requirements.txt requirements_tests.txt requirements_jax.txt /tmp
RUN python3 -m pip install \
  -r /tmp/requirements.txt \
  -r /tmp/requirements_tests.txt \
  -r /tmp/requirements_jax.txt
EOT
}

target "ray_tests" {
  platforms = ["linux/amd64"]
  contexts = {
    base_image = "target:jammy_python_sys"
  }
  context           = "${PROJECT_ROOT}"
  tags = [
    "${DOCKER_REPO}/tractorun-ray-tests:${DOCKER_TAG}"
  ]
  dockerfile-inline = <<EOT
FROM base_image
COPY requirements.txt requirements_tests.txt requirements_ray.txt /tmp
RUN python3 -m pip install \
  -r /tmp/requirements.txt \
  -r /tmp/requirements_tests.txt \
  -r /tmp/requirements_ray.txt
RUN apt install net-tools telnet -y
EOT
}

target "tensorproxy_tests" {
  platforms = ["linux/amd64"]
  contexts = {
    base_image = "target:jammy_python_sys"
  }
  context           = "${PROJECT_ROOT}"
  tags = [
    "${DOCKER_REPO}/tractorun-tensorproxy-tests:${DOCKER_TAG}"
  ]
  dockerfile-inline = <<EOT
FROM base_image
COPY requirements.txt requirements_tests.txt requirements_tensorproxy.txt /tmp
RUN python3 -m pip install \
  -r /tmp/requirements.txt \
  -r /tmp/requirements_tests.txt
COPY . /src
RUN python3 -m pip install -r "/tmp/requirements_tensorproxy.txt"
EOT
}

target "examples_runtime" {
  platforms = ["linux/amd64"]
  contexts = {
    base_image = "target:jammy_python_sys"
  }
  context           = "${PROJECT_ROOT}"
  tags = [
    "${DOCKER_REPO}/tractorun-examples-runtime:${DOCKER_TAG}"
  ]
  dockerfile-inline = <<EOT
FROM base_image
COPY requirements.txt requirements_tests.txt requirements_jax.txt requirements_examples.txt requirements_torch.txt /tmp
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cpu -r "/tmp/requirements_torch.txt"
RUN python3 -m pip install \
  -r /tmp/requirements.txt \
  -r /tmp/requirements_tests.txt \
  -r /tmp/requirements_jax.txt \
  -r /tmp/requirements_examples.txt
EOT
}
