ARG VARIANT="3.9-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

USER vscode

WORKDIR /workspace

ADD build/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN git init

ADD . .

USER root

RUN chown -R vscode:vscode /workspace

USER vscode

EXPOSE 8890:8890

CMD ["/workspace/start-jupyter-lab.sh" ]
