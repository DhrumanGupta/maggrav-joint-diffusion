#! /bin/bash

"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

source ~/.bashrc

wandb login