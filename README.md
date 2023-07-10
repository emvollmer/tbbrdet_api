# tbbrdet_api
[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/UC-emvollmer-tbbrdet_api/master)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/UC-emvollmer-tbbrdet_api/job/master)

Deepaas API for TBBRDet Model

To launch it, first install the package then run [deepaas](https://github.com/indigo-dc/DEEPaaS):
```bash
git clone https://github.com/emvollmer/tbbrdet_api
cd tbbrdet_api
source install_TBBRDet.sh 	# this sets up the venv with the required packages and installs the submodule TBBRDet as an editable project
pip install -e .
deepaas-run --listen-ip 0.0.0.0
```
The associated Docker container for this module can be found in https://github.com/emvollmer/DEEP-OC-tbbrdet_api.

## Project structure
```
├── LICENSE                <- License file
│
├── README.md              <- The top-level README for developers using this project.
│
├── requirements.txt       <- The requirements file for reproducing the analysis environment, e.g.
│                             generated with `pip freeze > requirements.txt`
│
├── setup.py, setup.cfg    <- makes project pip installable (pip install -e .) so
│                             tbbrdet_api can be imported
│
├── tbbrdet_api    <- Source code for use in this project.
│   │
│   ├── __init__.py        <- Makes tbbrdet_api a Python module
│   │
│   └── api.py             <- Main script for the integration with DEEP API
│
└── Jenkinsfile            <- Describes basic Jenkins CI/CD pipeline
```
