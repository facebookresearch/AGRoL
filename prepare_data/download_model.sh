#!/bin/sh
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

mkdir -p pretrained
cd pretrained/ || exit

# Download model command: coming soon

unzip agrol.zip
rm agrol.zip

printf "Pre-trained model was downloaded into pretreined/ folder!"
