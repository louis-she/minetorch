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

# install python dependencies
pip install -r requirements.txt

# init sqlite database
python3 cli.py db:init

# install frontend dependencies
cd web; yarn install

# start the development server
# this will start a flask web server in development mode and webpack --watch
cd ..; python cli.py dev

# visit localhost:5000
```

# Reset database

If any new fields are been added to ORM, you may want to execute the following commmand to reset the database.

```
python cli.py db:init
```

> This command will drop the table first and then create it. So the data in the table will be lossed. If you want to keep the data, just create the new column by sqlite plain sql.
