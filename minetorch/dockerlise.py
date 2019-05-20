
import os
from pathlib import Path

import docker

from minetorch.utils import make_runtime_dir, runtime_file

def build_image(experiment):
    # step 1 make Dockerfile
    snapshot = experiment.current_snapshot()
    experiment_dir = make_runtime_dir(experiment)
    runtime_dir = f'/{experiment.name}_{snapshot.id}'
    pip_dir = Path(__file__).parent
    with runtime_file(experiment_dir / 'Dockerfile', 'w') as f:
        content = f"""FROM python:3
WORKDIR {runtime_dir}
COPY . {runtime_dir}
RUN pip3 install git+https://github.com/minetorch/minetorch.git@develop
CMD ["python3", "run.py"]"""
        f.write(content)
    # step 2 build docker image
    client = docker.DockerClient(base_url='unix://var/run/docker.sock')
    img = client.images.build(path=experiment_dir.as_posix(), tag=f'{experiment.name}_{snapshot.id}')
    return img[0].tags[0]
