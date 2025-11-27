FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

COPY conda-lock.yml /tmp/conda-lock.yml

# Install conda-lock first
RUN mamba install -n base -c conda-forge conda-lock -y

# Install from lock file
RUN conda-lock install --name base /tmp/conda-lock.yml

# Clean up and fix permissions
RUN mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

WORKDIR /home/jovyan/work


