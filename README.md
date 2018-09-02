# Torchpack
Use Pytorch in a smart way.

## Quick Start

1. Clone this repo

```
git clone git@github.com:louis-she/torchpack.git
```

2. Goto examples and execute

```
python mnist.py
```

The `mnist.py` is pretty much like pytorch official mnist example https://github.com/pytorch/examples/blob/master/mnist/main.py, with some minor changes to be adapted to Torchpack.

This will create a `log` and `data` directory in working directory.

3. Execute tensorboard to visualize

```
tensorboard ./log
```

![](./images/plan-a.png)

4. Now let's change the loss function to cross_entropy(just follow some instructions in the comments of `mnist.py`), and train the mnist again.

![](./images/plan-a-b.png)