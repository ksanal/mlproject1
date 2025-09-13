# Use the Jenkins LTS image as the base
FROM jenkins/jenkins:lts

USER root

# Install prerequisites
RUN apt-get update -y && \
    apt-get install -y apt-transport-https ca-certificates curl gnupg software-properties-common && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add -

# Dynamically set Docker repo arch
RUN ARCH=$(dpkg --print-architecture) && \
    echo "deb [arch=${ARCH}] https://download.docker.com/linux/debian bullseye stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update -y && \
    apt-get install -y docker-ce docker-ce-cli containerd.io && \
    apt-get clean

# Add Jenkins to Docker group
RUN groupadd -f docker && \
    usermod -aG docker jenkins

# Prepare Docker storage
RUN mkdir -p /var/lib/docker
VOLUME /var/lib/docker

USER jenkins
