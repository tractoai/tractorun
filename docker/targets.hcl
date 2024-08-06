target "jammy_python_sys" {
  platforms = ["linux/amd64"]
  dockerfile-inline = <<EOT
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    python3 -m pip install --upgrade pip
EOT
}

target "torchesaurus_tests" {
  platforms = ["linux/amd64"]
  contexts = {
    base_image = "target:jammy_python_sys"
  }
  context           = "${PROJECT_ROOT}"
  tags = [
    "${DOCKER_REPO}/torchesaurus_tests:${DOCKER_TAG}"
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

target "torchesaurus_runtime" {  # TODO: find out if it is possible to reduce its size
  platforms = ["linux/amd64"]
  contexts = {
    base_image = "target:jammy_python_sys"
  }
  context           = "${PROJECT_ROOT}"
  tags = [
    "${DOCKER_REPO}/torchesaurus_runtime:${DOCKER_TAG}"
  ]
  dockerfile-inline = <<EOT
FROM base_image
COPY requirements.txt requirements_lightning.txt requirements_torch.txt /tmp
RUN python3 -m pip install -r /tmp/requirements_torch.txt
RUN python3 -m pip install \
  -r /tmp/requirements.txt \
  -r /tmp/requirements_lightning.txt
COPY . /src
RUN python3 -m pip install --no-deps "/src"
EOT
}

target "tractorax_runtime" {
  platforms = ["linux/amd64"]
  contexts = {
    base_image = "target:jammy_python_sys"
  }
  context           = "${PROJECT_ROOT}"
  tags = [
    "${DOCKER_REPO}/tractorax_runtime:${DOCKER_TAG}"
  ]
  dockerfile-inline = <<EOT
FROM base_image
COPY requirements.txt requirements_jax.txt /tmp
RUN python3 -m pip install \
  -r /tmp/requirements.txt \
  -r /tmp/requirements_jax.txt
COPY . /src
RUN python3 -m pip install --no-deps "/src"
EOT
}

target "tensorproxy_tests" {
  platforms = ["linux/amd64"]
  contexts = {
    base_image = "target:jammy_python_sys"
  }
  context           = "${PROJECT_ROOT}"
  tags = [
    "${DOCKER_REPO}/tensorproxy_tests:${DOCKER_TAG}"
  ]
  dockerfile-inline = <<EOT
FROM base_image
COPY requirements.txt requirements_tests.txt requirements_tensorproxy.txt /tmp
RUN python3 -m pip install \
  -r /tmp/requirements.txt \
  -r /tmp/requirements_tests.txt
COPY . /src
RUN python3 -m pip install --extra-index-url https://artifactory.nebius.dev/artifactory/api/pypi/nyt/simple -r "/tmp/requirements_tensorproxy.txt"
RUN python3 -m pip install --no-deps "/src"
EOT
}

target "generic_runtime" {
  platforms = ["linux/amd64"]
  contexts = {
    base_image = "target:jammy_python_sys"
  }
  context           = "${PROJECT_ROOT}"
  tags = [
    "${DOCKER_REPO}/generic_runtime:${DOCKER_TAG}"
  ]
  dockerfile-inline = <<EOT
FROM base_image
COPY requirements.txt /tmp
RUN python3 -m pip install \
  -r /tmp/requirements.txt
COPY . /src
RUN python3 -m pip install --no-deps "/src"
EOT
}

target "demo_image" {
  platforms = ["linux/amd64"]
  tags = [
    "${DOCKER_REPO}/demo:${DOCKER_TAG}"
  ]
  dockerfile-inline = <<EOT
FROM quay.io/jupyter/pytorch-notebook:cuda12-python-3.11.8

RUN pip3 install torchvision wandb

COPY . /src
USER root
RUN python3 -m pip install "/src"
USER 1000

EOT
}

target "papyrax_demo" {
  platform = ["linux/amd64"]
  tags = [
    "${DOCKER_REPO}/papyrax_demo:${DOCKER_TAG}"
  ]
  dockerfile-inline = <<EOT
FROM cr.ai.nebius.cloud/crnf2coti090683j5ssi/ytsaurus/papyrax_main:2024-07-11-03

RUN ln -fns /usr/bin/python3.10 /usr/bin/python3

COPY . /src
RUN pip install /src
EOT
}
