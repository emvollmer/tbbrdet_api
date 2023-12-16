#!/bin/bash

git submodule init && git submodule update --remote --merge && pip3 install -e ./TBBRDet && pip3 install -e . && pip3 install -r requirements-test.txt
