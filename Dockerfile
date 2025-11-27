FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

COPY conda-lock.yml /tmp/conda-lock.yml

# Install conda-lock and use render to install packages
RUN mamba install -n base -c conda-forge conda-lock -y && \
    conda-lock render -p linux-64 /tmp/conda-lock.yml --filename-template /tmp/env-{platform}.lock && \
    mamba update --name base --file /tmp/env-linux-64.lock --yes && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

WORKDIR /home/jovyan/work


