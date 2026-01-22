#! /bin/bash

conda install conda-forge::google-cloud-sdk -y

conda install -c conda-forge crcmod -y

gcloud auth login

gcloud config set project cedar-card-482809-c3