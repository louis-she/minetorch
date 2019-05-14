
import os
import docker
from datetime import datetime

def init_image_dir():
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

def build_image(runtime_dir):
    #step 1 创建Dockerfile
    docker_file = os.path.join(os.getcwd(), 'Dockerfile')
    #删除已经存在的Dockerfile
    if os.path.exists(docker_file):
        os.remove(docker_file)

    with open(docker_file, 'w') as f:
        f.write("FROM python:3\n\n")
        f.write("WORKDIR /expirment\n\n")
        # mnist_file = os.path.join(os.getcwd(), "../../examples/mnist.py")
        # print(mnist_file)
        f.write("COPY . /expirment\n\n")
        # require_file = os.path.join(os.getcwd(), "../requirement.txt")
        f.write("RUN pip3 install -r requirements.txt\n\n")
        # f.write("CMD ['python3', 'mnist.py']")

    # with open(docker_file, 'r') as f:
    #     client = docker.DockerClient(base_url='unix://var/run/docker.sock')
    #     tagtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     client.images.build(path= os.getcwd())

