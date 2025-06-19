# MAMA-MIA-Challenge

This repository collects the preprocessing, training and inference code for our MAMA-MIA challenge submission.

## Set up training dependencies
Install your favorite python package manager (e.g. miniconda).

```bash
conda create -n 'mamamia' python==3.12.9
conda activate mamamia
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Test inference Docker container
```bash
docker run -it --rm --entrypoint /bin/bash --gpus all --shm-size=1024m `
--mount type=bind,source=<this_repository_checkout_directory>/mama-mia-challenge/inference,target=/app/ingested_program `
--mount type=bind,source=<test_data_directoy>,target=/dataset `
--mount type=bind,source=<test_output_directory>,target=/output `
lgarrucho/codabench-gpu:latest `
 -c "python /app/ingested_program/test_model.py /dataset /output"
```

