> This document is for developers, if you just want to extend minetorch(add components to minetorch), follow [extend.md](extend.md)


# Setup development environment

Minetorch is totally python3(not consider python2 at all), so before starting please install python3 with ssl module enabled.

After python3 installed, follow the next steps to setup Minetorch development environment:

```shell
# clone this repo, make sure you have access, or you should fork and clone your own.
git clone git@github.com:minetorch/minetorch.git

# navigate to minetorch source code dir
cd minetorch/minetorch

# enable python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install libomp for PyTorch
brew install libomp

# install python dependencies
pip3 install -r requirements.txt

# init sqlite database
python3 cli.py db:init

# install frontend dependencies
cd web; yarn install

# start the development server
# this will start a flask web server in development mode and webpack --watch
cd ..; python3 cli.py dev

# visit localhost:3100
```

# Reset database

If any new fields are been added to ORM, you may want to execute the following commmand to reset the database.

```
python3 cli.py db:init
```

> This command will drop the table first and then create it. So the data in the table will be lossed. If you want to keep the data, just create the new column by sqlite plain sql.

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
