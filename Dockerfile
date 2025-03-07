# Use a base image with Miniconda
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy only the environment.yml file into the container
COPY environment.yml /app/environment.yml

# Install dependencies using Conda
RUN conda env create -f /app/environment.yml && \
    conda clean --all -y

# Set the default shell to use Conda
SHELL ["/bin/bash", "-c"]

# Activate the environment by default
RUN echo "conda activate your_env_name" >> ~/.bashrc

# Start with bash
CMD ["bash"]
