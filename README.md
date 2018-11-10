![](./images/minetorch.jpg)
# Minetorch

In [Minecraft](https://minecraft.net/), torches are very important for mining. No one can get all the diamonds without torch. So is data-mining, A special torch named [pytorch](http://pytorch.org/) can help us get the dimonds in data. Minetorch is a tools collection for miners, to use pytorch in a more convinent way.

## Installation

```
pip install minetorch
```

## Quick Start

1. Clone this repo

```
git clone git@github.com:louis-she/minetorch.git
```

2. Goto examples and execute

```
python mnist.py
```

The `mnist.py` is pretty much like pytorch official mnist example https://github.com/pytorch/examples/blob/master/mnist/main.py, with some minor changes to be adapted to Minetorch.

This will create a `log` and `data` directory in working directory.

3. Execute tensorboard to visualize

```
tensorboard ./log
```

![](./images/plan-a.png)

4. Now let's change the loss function to cross_entropy(just follow some instructions in the comments of `mnist.py`), and train the mnist again.

![](./images/plan-a-b.png)