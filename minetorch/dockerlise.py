
import os, stat
import docker
from datetime import datetime
import docker

def check_image_dir():
    """在用户主目录创建文件夹用于存储dockerimage
    """
    os.environ['HOME']
    os.path.expandvars('$HOME')
    originalpath = os.path.expanduser('~') + "/.minetorch"
    folder = os.path.exists(originalpath)
    if not folder:
        os.makedirs(originalpath)
        print("~~~ make " + originalpath + " OK ~~~")
    else:
        print("orignial dir already exists")
    return originalpath
                                

def build_image(experiment_name, snapshot_id):
    #step 1 创建Dockerfile
    path = check_image_dir()
    runtime_dir = f'/{experiment_name}_{snapshot_id}'
    image_dict = path + runtime_dir
    
    docker_file = os.path.join(image_dict, 'Dockerfile')
    #删除已经存在的Dockerfile
    if os.path.exists(docker_file):
        os.chmod(docker_file, 0777)
        os.remove(docker_file)
    # os.makefile(docker_file)
    with open(docker_file, 'w') as f:
        f.write("FROM minetorch_base_image\n")
        f.write(f'COPY {image_dict} {runtime_dir}\n')
        f.write(f'EXEC CD {runtime_dir}; python3 run.py')
    client = docker.DockerClient(base_url='unix://var/run/docker.sock')
    return client.image.build(path=os.getcwd())

