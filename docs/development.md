> This document is for developers, if you just want to extend minetorch(add components to minetorch), follow [extend.md](extend.md)


# Setup development environment

Minetorch is totally python3(not consider python2 at all), so before starting please install python3 with ssl module enabled.

After python3 installed, follow the next steps to setup Minetorch development environment:

```shell
# clone this repo, make sure you have access, or you should fork and clone your own.
git clone git@github.com:minetorch/minetorch.git

# navigate to minetorch source code dir
cd minetorch

# enable python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install libomp for PyTorch
brew install libomp

# install python dependencies
pip3 install -r requirements.txt

# init sqlite database
python3 minetorch/cli.py db:init

# install frontend dependencies
cd minetorch/web; yarn install

# start the development server
# this will start a flask web server in development mode and webpack --watch
cd ../..; python3 minetorch/cli.py dev

# visit localhost:3100
```

# Test training process

When you have used the web dashboard to generate a training config file under `~/.minetorch`, you have 2 choices to test the training process,

1. Install your local version of Minetorch and then run the `run.py` directly
```
# Goto the minetorch base dir
pip install .
cd ~/.minetorch/${your_experiment_name}_${snapshot_id}; python run.py
```

2. Just using `cli.py`
```
python cli.py runtime:run --config $absolute_path_of_a_config_file
```

# About logging

Since users can view and filter the log in the web dashboard. So it's really important to decide how and what for logging. There are together 5 level of logging, the following table shows what does they mean and how to use it,

| Level   |  Description    |
|----------|:-------------|
| Debug | For developer only. By default, `python3 cli.py dev` will set the log level to this |
| Info | For all users. The information should let users know what is happenning. By default, `python3 cli.py prod` will set the log level to this |
| Warn | For all users. |
| Error | For all users. |

# Reset database

If any new fields are been added to ORM, you may want to execute the following commmand to reset the database.

```
python3 cli.py db:init
```

> This command will drop the table first and then create it. So the data in the table will be lossed. If you want to keep the data, just create the new column by sqlite plain sql.

# Generate python proto files

If you have changed any files in `protos` folder, you should use the following command to regenerate the Python proto files.

```
python3 cli.py proto:compile
```

# API

APIs are defined and well documented [here](../minetorch/web/api/__init__.py). We follow the Rails Resource like design pattern. Think about resource `Book`, there should be at most 7 APIs to reach the `Book` resource:

| Method   |    Path       |  Description |
|----------|:-------------| :-------------|
| GET |  /books | show the indices of all the books |
| GET | /books/new | show the page of book creating form |
| POST | /books | create a new book |
| GET | /books/:id | get the detail of the book with the specified `:id` |
| GET | /books/:id/edit | show the page of book editing form with the specified `:id` |
| PATCH/PUT | /books/:id | update the book with the specified `:id` |
| DELETE | /books/:id | delete the book with the specified `:id` |

Agreement of the response:

| Scenario |  Response Code |
|----------|:-------------|
| Everything goes well | 200 |
| The data submitted by the user is not valid | 422 |
| The data submitted by the user is violated the uniq constraint | 409 |
| Server Error | 5XX |

The frontend should take different actions for these response code, and trigger an `unknown error` if the response code is not been defined here.
