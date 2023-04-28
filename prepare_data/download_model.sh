#!/bin/sh
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

# Download model command
wget https://github.com/facebookresearch/AGRoL/releases/download/v0/agrol_AMASS_pretrained_weights.zip

unzip agrol_AMASS_pretrained_weights.zip -d pretrained_weights
rm agrol_AMASS_pretrained_weights.zip

echo "Pre-trained model was downloaded into './pretrained_weights' folder!"
