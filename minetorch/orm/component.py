from .base import Base
from peewee import CharField, IntegerField, DateTimeField, TextField

class Component(Base):
    name = CharField(unique=True)
    category = CharField()
    settings = TextField()

    def create(self, **query):
        if 'category' not in query:
            query['category'] = self.__class__

class Model(Component):
    category = CharField(default='Model')
    class Meta:
        table_name = 'component'

class Dataset(Component):
    category = CharField(default='Dataset')
    class Meta:
        table_name = 'component'

class Dataflow(Component):
    category = CharField(default='Dataflow')
    class Meta:
        table_name = 'component'

class Optimizer(Component):
    category = CharField(default='Optimizer')
    class Meta:
        table_name = 'component'

class Loss(Component):
    category = CharField(default='Loss')
    class Meta:
        table_name = 'component'
