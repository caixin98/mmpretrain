#for every config like configs/a_optical_face/optical_mobilefacenet_celeb/kernel_size=*.py
#do scripts/dist_train.sh $config

#!/bin/bash
tsp -S 1

find configs/a_optical_face/vit_optical/model.backbone.optical.mask_kernel_shape=full_gaussian -name '*.noise_ratio*' -type f -print0 |
while IFS= read -r -d '' file; do
    # Perform your action on each file (e.g., execute a script)
    tsp ./scripts/dist_train.sh $file 
    # Or perform any other desired action
    # For example, echo the file path
    echo "Processing file: $file"

done