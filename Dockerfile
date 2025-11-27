FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

COPY environment.yml /tmp/environment.yml

# Install packages from environment.yml
RUN mamba env update -n base -f /tmp/environment.yml && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

WORKDIR /home/jovyan/work


