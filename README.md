# Minetorch

![](./images/minetorch.jpg)

In [Minecraft](https://minecraft.net/), torches are very important for mining. No one can get all the diamonds without a torch. So is data-mining, A special torch named [PyTorch](http://pytorch.org/) can help us get the diamonds in data. Minetorch is a tools collection for miners, to use PyTorch in a more convenient way.

## Features

* Logging of course.
* Play with Google Sheets.
* Visualization. built-in with Tensorboard or Matplotlib.
* Resumption. Default will resume from the last checkpoint.
* Many hook points.
* Training code. Or actually a namespace, used to easily sperate different training status.

## Installation

```
pip install minetorch
```

## Minetorch X Google Sheets

Minetorch can easily work with Google Sheets. To step up, you should have a Google account and, if you are in China, a proxy.

This is the [demo sheet](https://docs.google.com/spreadsheets/d/1SkS1NWdn1gyrSTbtDRCFUeEsE8dHBJkH0W391bOIGB4/edit#gid=0) that produced with Minetorch training on MNIST dataset.

![](./images/sheet.png)

**Enable Google API**


1. Sign in to https://console.developers.google.com with your Google account
2. Create a new `Project`
3. In the API Libarary, search `drive` and `sheet`, and enable them both. During this, Google should have alert you about creating the service account. If you have get the right credential file, ignore step 4.
4. Create a Google Service Account in at https://console.developers.google.com/iam-admin/iam .

After all these steps you should get a `json` credential file which looks like this:

```
{
  "type": "xxxxxx",
  "project_id": "xxxxxx",
  "private_key_id": "xxxxxx",
  "private_key": "-----BEGIN PRIVATE KEY-----\n .......... \n-----END PRIVATE KEY-----\n",
  "client_email": "xxxxxx@xxxxxx-xxxxxxxxxx.iam.gserviceaccount.com",
  "client_id": "xxxxxxxxxxxxxxxxxxxxx",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.iam.gserviceaccount.com"
}
```

**Create the spreadsheet**

1. At http://drive.google.com/, create a new spreadsheet.
2. Share that spreadsheet with your service account, you can find your service account email address in your credential file.
3. Get the id of the spreadsheet, it's right in the middle of the URL.

**Ready to go**

In your code, just create a `minetorch.spreadsheet.GoogleSheet` instance and pass it to `Miner`.

```python
miner = Miner(
    # ...
    sheet=GoogleSheet('Your sheet id', 'path-of-your-google-service-account-credensial-file.json')
    # ...
)

miner.train()
```

and that's it, your spreadsheet should be automatically updated when you start your training process.

> Minetorch update the sheet in async way, so it will not slow down the training speed.

## Quick Start

1. Clone this repo

```
git clone git@github.com:louis-she/minetorch.git
```

2. Goto examples and execute

```
python mnist.py
```

The `mnist.py` is pretty much like PyTorch official mnist example https://github.com/pytorch/examples/blob/master/mnist/main.py, with some minor changes to be adapted to Minetorch.

This will create an `alchemistic_directory` and `data` directory in the working directory.

3. Execute tensorboard to visualize

```
tensorboard ./log
```

![](./images/plan-a.png)

or if you used `matplotlib` as the drawer, there would be 2 images under the path `${alchemistic_directory}/${code}/graph` named loss.png and accuracy.png

loss.png             |  accuracy.png
:-------------------------:|:-------------------------:
![](./images/loss.png)| ![](./images/accuracy.png)



> Note: Since PyTorch 1.0 comes with CUDA10 but the newest version of Tensorboard just supported CUDA9, so if you are using Tensorboard as a drawer, make sure you got CUDA9 installed, or it's better to use matplotlib. Due to this, Minetorch default drawer has changed to matplotlib instead of Tensorboard.

4. Now let's change the loss function to cross_entropy(just follow some instructions in the comments of `mnist.py`), and train the mnist again.

![](./images/plan-a-b.png)

## How it works

Minetorch lets users focusing on **the only** necessary things, and will take care of the others. It's more like a skeleton and users should provide necessary components to make it work.

The parameters users must provide:

| parameters | type | description |
| ---------- | ---- | ----------- |
| alchemistic_directory | string | path of a directory, all the `checkpoint`, `log` or `graph` will be saved in this `alchemistic_directory` |
| train_dataloader | torch.utils.data.DataLoader | Used to tell minetorch how to load training data |
| model | torch.nn.Module | PyTorch's nn.Module |
| loss_func | callable | A special hook function, should receive 2 arguments: `data` which yields by the loader and `trainer` which is the trainer instance, this function should return a single scalar which is the loss |

And that's it, minetorch will take care of other things like `logging`, `resuming`, `visualization` etc... The names of the components are actually the parameters of the `Trainer` class. The other optional parameters are:

| parameters | type | description |
| ---------- | ---- | ----------- |
| val_dataloader | torch.utils.data.DataLoader | Used to tell minetorch how to load validation data |
| resume | bool or string  | Defaults to True, means the minetorch will try to resume from the latest checkpoint. If a string is given then minetorch will resume from a specified checkpoint, the string could be number of epoch, name of the checkpoint file or absolute path of the checkpoint file, false means train from scratch|
| eval_stride  | int | Defaults to 1, how many epochs to run a validation process |
| persist_stride  | int | Defaults to 1, how many epochs to save a checkpoint |
| drawer  | minetorch.Drawer or string | Defaults to 'matplotlib'. To generate graphs of the whole training process, now support 'tensorboard' and 'matplotlib', can also write a customized Drawer by yourself |
| code  | string | it's actually a subdirectory path of `alchemistic_directory`, for separating the results of different attempts, every attempt should have a unique name |
| hooks | dict | Defining hook function, see [Hooks](#hooks) |
| max_epochs | int  | How many epochs to train, defaults to None, means unlimited |
| logging_format | string  | Defaults to '%(levelname)s %(asctime)s %(message)s', same as logging's format |
| trivial | bool  | Defaults to False, if set to True, both training and validation process will be broken in 10 iterations, useful at development stage |
| in_notebook | bool  | Defaults to False, if use minetorch in jupyter notebook environment, set this option to True to have a better output |
| statable | dict  | Defaults to {}. A statble is an object which has implemented `state_dict` and `load_state_dict`, pass these objects in statable dict will let minetorch handle the load and save operation on it. For instance, Learning Rate Schedular is a typical statable object. Don't put optimizer and model here, minetorch already know that they are statble.|

## Hooks

Minetorch provided many hook points for users to control the training behaviors. All the hook functions receive current  `trainer` instance as the arguments so the hook function can access all the functions and status of `Trainer`.

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

## TODOs

- [x] More hook points.
- [ ] Number of x-axis of Drawer generated images are not changed after resuming from a checkpoint.
- [ ] Drawer DB Adapter. To persist drawer data to DB and then can be visualization by any tools.
- [ ] Graceful exists.
- [ ] Dockerize.

