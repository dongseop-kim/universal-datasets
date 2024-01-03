#!/bin/bash

# google drive link
# https://drive.google.com/file/d/1_IodNv7NLbXKjE5MyB5WY8zGCtPyQ8Vp/view?usp=sharing
file_id="1_IodNv7NLbXKjE5MyB5WY8zGCtPyQ8Vp"
output_file="PASCAL-VOC.zip"

gdown https://drive.google.com/uc?id=${file_id} -O ${output_file}

# Unzip the file
unzip ${output_file}

# Remove the .zip file after extraction (optional, you can remove this line if you want to keep the .zip file)
rm ${output_file}