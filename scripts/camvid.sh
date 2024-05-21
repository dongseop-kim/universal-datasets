#!/bin/bash

# google drive link
# https://drive.google.com/file/d/15e7J7bLBosM8Aqb6LtkbD7gQFzbZ9TbY/view?usp=sharing
file_id="15e7J7bLBosM8Aqb6LtkbD7gQFzbZ9TbY"
output_file="CamVid.zip"

gdown https://drive.google.com/uc?id=${file_id} -O ${output_file}

# Unzip the file
unzip ${output_file}

# Remove the .zip file after extraction (optional, you can remove this line if you want to keep the .zip file)
rm ${output_file}