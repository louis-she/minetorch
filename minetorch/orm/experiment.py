from .base import Base
from peewee import CharField, IntegerField, DateTimeField

class Experiment(Base):
    name = CharField(unique=True)
    total_training_time = IntegerField(default=0)
    is_training = IntegerField(default=0, null=True)
    last_stopped_at = DateTimeField(null=True)
