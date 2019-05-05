from .base import Base
from peewee import Field, CharField, IntegerField, DateTimeField

class Snapshot(Base):
    name = CharField(unique=True)
    total_training_time = IntegerField(default=0)
    stopped_at = DateTimeField(null=True)
    category = IntegerField()

    def clone(self):
        pass
