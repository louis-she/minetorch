![](./images/minetorch.jpg)

# Minetorch

In [Minecraft](https://minecraft.net/), torches are very important for mining. No one can get all the diamonds without torch. So is data-mining, A special torch named [pytorch](http://pytorch.org/) can help us get the dimonds in data. Minetorch is a tools collection for miners, to use pytorch in a more convinent way.

## How it works

Minetorch let users focusing on **the only** necessary things, and will take care of the others. It's more like a skeleton and users should provide necessary components to make it work.

The components users must provided:

| components | type | description |
| ---------- | ---- | ----------- |
| alchemistic_directory | string | path of a directory, all the `checkpoint`, `log` or `graph` will be saved in this `alchemistic_directory` |
| train_dataloader | torch.utils.data.DataLoader | Used to tell minetorch how to load training data |
| model | torch.nn.Module | Pytorch's nn.Module |
| loss_func | callable | A special hook function, should receive 2 arguments: `data` which yields by the loader and `trainer` which is the trainer instance |

And that's it, minetorch will take care of others things like `logging`, `resumming`, `visualization` etc... The names of the components are actually the parameters of the `Trainer` class. The other optional parameters are:

| components | type | description |
| ---------- | ---- | ----------- |
| val_dataloader | torch.utils.data.DataLoader | Used to tell minetorch how to load validation data |
| resume | bool or string  | Defaults to True, means the minetorch will try to resume from the latest checkpoint, if a string is given then minetorch will resume from a specified checkpoint, the string could be number of epoch, name of the checkpoint file or absolute path of the checkpoint file, false means train from scratch|
| eval_stride  | int | Defaults to 1, how many epochs to run a validation process |
| persist_stride  | int | Defaults to 1, how many epochs to save a checkpoint |
| drawer  | minetorch.Drawer or string | To generate graphs of the whole training process, now support 'tensorboard', can also write a customized Drawer |
| code  | string | it's actually a sub directory path of `alchemistic_directory`, for sperating the results of different attempts, every attempts should have a uniq name |
| hooks | dict | Defining hook function, see [Hooks](#hooks) |
| max_epochs | int  | How many epochs to train, defaults to None, means unlimited |
| logging_format | string  | Defaults to '%(levelname)s %(asctime)s %(message)s', same as logging's format |

## Hooks

Minetorch provided many hook points for users to controller the training behaviors. All the hook function receive currnet `trainer` instance as the arguments so the hook function can access all the functions and status of `Trainer`.

| hook points | description |
| ----------- | ----------- |
| after_init | called after the construct function of Trainer been called |
| before_epoch_start | called before every epoch started |
| after_epoch_end | called after every epoch ended |
| before_train_iteration_start | called before every training iteration started |
| after_train_iteration_end  | called after every training iteration ended |
| before_val_iteration_start | called before every validation iteration started |
| after_val_iteration_end  | called after every validation iteration ended |
| before_checkpoint_persisted | called before checkpoint persisted |
| after_checkpoint_persisted | called after checkpoint persisted |
| before_quit | called before the max_epochs exceeded and about to quit training |

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