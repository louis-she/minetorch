
import docker
from minetorch.utils import make_runtime_dir, runtime_file
import socket
import time

def docker_build(experiment):
    # step 1 make Dockerfile
    snapshot = experiment.current_snapshot()
    experiment_dir = make_runtime_dir(experiment)
    runtime_dir = f'/{experiment.name}_{snapshot.id}'
    with runtime_file(experiment_dir / 'Dockerfile', 'w') as f:
        content = f"""FROM python:3
WORKDIR {runtime_dir}
COPY . {runtime_dir}
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple git+https://github.com/minetorch/minetorch.git@develop
CMD ["python3", "run.py"]"""
        f.write(content)
    # step 2 build docker image
    print("step 2 begin")
    client = docker.DockerClient(base_url='unix://var/run/docker.sock')
    img = client.images.build(path=experiment_dir.as_posix(), tag=f'{experiment.name}_{snapshot.id}')
    docker_tag = img[0].tags[0]
    # step 3 run local registry
    print("step 3 begin")
    client.containers.run(image='docker.io/registry:latest', detach=True, ports={'5000/tcp': 6000})
    # step 4 tag image to local registry
    print("step 4 begin")
    local_ip = socket.gethostbyname(socket.gethostname())
    local_tag = f'{local_ip}:6000/{docker_tag}'
    img[0].tag(repository=local_tag)
    # step 5 push image to local registry
    print("step 5 begin")
    time.sleep(10)
    client.images.push(repository=local_tag)
    docker_command = f'docker pull {local_tag}'
    return docker_command