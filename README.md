# Artifact for the Tool Paper “MoGym: Using Formal Models for Training and Verifying Decision-making Agents”


## Abstract

MoGym is an integrated toolbox enabling the training and verification of machine-learned decision-making agents based on formal models in the JANI-model format. This artifact for our tool paper “MoGym: Using Formal Models for Training and Verifying Decision-making Agents” demonstrates all features of MoGym and allows reproducing all the results presented in the paper. Starting from a formal model in JANI-model format, we demonstrate how a decision-making agent in the form of a neural network (NN) can be learned using MoGym as a training environment (cf. Sect. 2 of the paper). Technically speaking, the NN resolves non-determinism of one of the automata of the model. Afterwards, we show how the quality of the formerly learned NN can be assessed with MoGym leveraging our extension of the statistical model checker `modes` (cf. Sect. 3 in the paper). Finally, we provide all the models and infrastructure necessary to reproduce our experimental insights (cf. Sect. 4 in the paper).

The artifact is available [online on Zenodo](https://doi.org/10.5281/zenodo.6510840).

The artifact is provided as a Docker image (platform `linux/amd64`). It does not require any special hardware besides a fairly modern system with at least 8 GB of main memory (lower may work but is not recommended).
Most experiments will work with 8 GB of RAM but for the largest learning tasks more would be better.

We aim at a reusable badge: MoGym is integrated in Momba and we expect someone who wants to reuse MoGym to use the latest version of Momba available via the Python Package Index. Detailed installation instructions and a documentation how to use Momba are [available online](https://momba.dev/). This documentation also covers the [MoGym part of Momba](https://momba.dev/gym/) and explains how everything can be used beyond the paper. The source code of the MoGym parts implemented in Momba is available on [GitHub](https://github.com/koehlma/momba/tree/main/momba/gym) and in the directory `/home/vscode/.local/lib/python3.9/site-packages/momba/gym` in the provided Docker image.

Our extensions of `modes` have not been integrated into the official release of the Modest Toolset yet, but an integration is planned. Until this is done, one needs to unpack the `modest` executable from this artifact (in the `vendor` directory) and make sure it is in the `PATH` (i.e., found when executing `modest`). The Momba `gym` API should then work out-of-the-box.

- Licenses: Momba itself is available [under MIT on GitHub](https://github.com/koehlma/momba). Our extension of `modes` is licensed under the same license as the Modest Toolset (see `vendor/Modest/License-Modest-Toolset.txt`). This license allows the unlimited usage for any non-commercial applications.

- Libraries and Dependencies: The dependencies and used libraries are documented in `pyproject.toml` as per Python standard [PEP 621](https://peps.python.org/pep-0621/). Also, Momba should work with the latest versions of all its dependencies.

This artifact can also be used outside of Docker assuming some basic knowledge about Python packaging and virtual environments. We are using [Poetry](https://python-poetry.org) for packaging and dependency management. Hence, it suffices to extract the files of this artifact (from the `/workspace` directory of the provided Docker image or fetch them from our [GitHub repository](https://github.com/udsdepend/cav22-mogym-artifact)) and then run `poetry install` to obtain a virtual environment with everything necessary to run Jupyter Lab and the notebooks provided for our experiments.


## Content of the Artifact

Besides this README, the artifact contains a Docker image in the form of the file `cav22-mogym.image.tar` exported with `docker save`. It also contains the `vendor` directory with the modified version of the Modest Toolset and a `LICENSE` file with additional license information.

Please follow the instructions below to load the Docker image into your system and afterwards start the web interface of the artifact.

Within the Docker image, you will find the following:

- The `experiments` directory contains the learned NNs for `cdrive.2`, `Racetrack`, `elevators`, and `firewire` we used in the paper. With those files, it is possible to obtain the exact same NNs we used for our paper. In fact, the artifact loads the NNs from these files.

- The `jani-models` directory contains additional MDP QVBS models on which MoGym can be in principle applied. As noted in the paper, however, learning a NN for these models may not be successful.

- The `mogym` directory contains the Python and [RLMate](https://pypi.org/project/rlmate/) scripts used for learning as well as the Jupyter notebook `Experiments.ipynb` which we provide to reproduce the results presented in the  paper.

- The `tracks` directory contains additional track files for the Racetrack benchmark.

- The `vendor` directory contains the modified version of the Modest Toolset which is necessary for DSMC.

- The `paper.pdf` file is the paper this artifact belongs to.

In addition, there are some other configuration files for Jupyter, the Python project and the `Dockerfile` used to obtain the container.


## Using the Artifact

To use the artifact, you first need to load the Docker image from the provided `.tar` file:
```
docker load --input cav22-mogym.image.tar
```
Afterwards, you can start Jupyter Lab, embedded in the container, with:
```
docker run --platform linux/amd64 -p 8890:8890 --rm -it cav22-mogym
```
This will expose a web interface on port `8890` on the machine Docker is running on. On this machine, you can connect to the web interface in a web browser by opening `http://localhost:8890/lab/tree/mogym/Experiments.ipynb`. The web interface will show a file explorer (left sidebar) for exploring the artifact and the main notebook with the experiments (`mogym/Experiments.ipynb`).

Please note, if you are running Docker inside of WSL 2 on Windows (not using Docker for Windows), the container listens on port `8890` within WSL 2, however, this port may not be accessible from a web browser on the Windows host (as WSL 2 is an isolated virtual machine). You can try to connect to the IP of WSL 2 directly but we recommend using either a proper installation of Docker for Windows or a Linux system.


## Resource Requirements

The artifact has been tested on an Intel Core i7-4790CPU @ 3.60GHz with 32GB RAM, 8 cores (with hyper threading), and Ubuntu 20.04.