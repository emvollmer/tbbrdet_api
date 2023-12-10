# tbbrdet_api
[![Build Status](https://jenkins.services.ai4os.eu/buildStatus/icon?job=AI4OS-HUB-TEST%2Ftbbrdet_api%2Fmaster)](https://jenkins.services.ai4os.eu/job/AI4OS-HUB-TEST/job/tbbrdet_api/job/master/)

DEEPaaS API for TBBRDet Model

To launch it, first install the package via the provided bash scripts, then run [deepaas](https://github.com/ai4os/DEEPaaS):
```bash
wget https://raw.githubusercontent.com/emvollmer/tbbrdet_api/master/deployment_setup.sh
source deployment_setup.sh 	# this sets up the deployment (CUDA, CUDNN, Python3.6)
source install_TBBRDet.sh 	# this sets up the venv with all required packages and installs the both API and submodule TBBRDet as editable
deep-start
# Alternatively
deepaas-run --listen-ip 0.0.0.0
```
When re-deploying after initial setup, remember to activate the virtual environment before running deepaas:
```bash
source venv/bin/activate
deep-start
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
├── data           <- Folder to download data to
│
├── models         <- Folder to save trained or downloaded models to
│
├── tbbrdet_api    <- Source code for the API to integrate the submodule TBBRDet with the platform.
│   │
│   ├── __init__.py        <- Makes tbbrdet_api a Python module
│   │
│   └── api.py             <- Main script for the integration with DEEP API
│   │
│   └── fields.py          <- Schema for frontend via Swagger UI
│   │
│   └── misc.py            <- Script containing helper functions
│
└── Jenkinsfile            <- Describes basic Jenkins CI/CD pipeline
```
