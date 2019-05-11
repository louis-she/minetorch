import peewee
from peewee import SqliteDatabase, Field, TimestampField,\
        CharField, IntegerField, DateTimeField, TextField, ForeignKeyField
from peewee import Model as PeeweeModel
from playhouse.shortcuts import model_to_dict
import datetime
import logging
import json

logger = logging.getLogger('peewee')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

db = SqliteDatabase('minetorch.db')

class JsonField(Field):
    field_type = 'json'

    def db_value(self, value):
        if isinstance(value, dict):
            return json.dumps(value)
        return value

    def python_value(self, value):
        if isinstance(value, str):
            return json.loads(value)
        return value


class Base(PeeweeModel):
    created_at = DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    updated_at = TimestampField()
    deleted_at = DateTimeField(null=True)

    def to_json_serializable(self):
        return model_to_dict(self)

    def delete(self):
        self.deleted_at = datetime.datetime.now(datetime.timezone.utc)
        self.save()

    class Meta:
        database = db
        legacy_table_names = False


class Experiment(Base):
    name = CharField(unique=True)
    total_training_time = IntegerField(default=0)
    is_training = IntegerField(default=0, null=True)
    last_stopped_at = DateTimeField(null=True)

    def create_draft_snapshot(self):
        """Try to create a draft snapshot, if there's already a draft version,
        should raise an error, otherwise it should check if there already have an
        current version, if yes, then clone it and return, if no, create a brand new
        snapshot(this is offen right after the Experiment is been created)
        """
        draft_snapshot = self.draft_snapshot()
        if draft_snapshot:
            raise 'There is already a draft snapshot'
        current_snapshot = self.current_snapshot()
        if current_snapshot:
            return current_snapshot.clone()
        else:
            return Snapshot.create(experiment_id=self.id)

    def draft_snapshot(self):
        try:
            return self.snapshots.where(Snapshot.category == 0).get()
        except peewee.DoesNotExist:
            return None

    def current_snapshot(self):
        try:
            return self.snapshots.where(Snapshot.category == 1).get()
        except peewee.DoesNotExist:
            return None


class Snapshot(Base):
    total_training_time = IntegerField(default=0)
    stopped_at = DateTimeField(null=True)
    # 0: draft 1: current 2: archived
    category = IntegerField(default=0)
    code = TextField(null=True)
    experiment = ForeignKeyField(Experiment, backref='snapshots')

    def clone(self):
        pass

    def is_draft(self):
        return self.category == 0

    def is_current(self):
        return self.category == 1

    def is_archived(self):
        return self.category == 2


class Component(Base):
    name = CharField()
    category = CharField()
    settings = JsonField(null=True)
    snapshot = ForeignKeyField(Snapshot, backref='components')
    code = TextField(null=True)

    class Meta:
        indexes = (
            (('name', 'snapshot'), True),
        )

    @classmethod
    def select(cls):
        return super().select().where(cls.category == str(cls.__name__))

    @classmethod
    def create(cls, **query):
        if 'category' not in query:
            query['category'] = cls.__name__
        return super().create(**query)


class Model(Component):
    category = CharField(default='Model')
    snapshot = ForeignKeyField(Snapshot, backref='models')
    class Meta:
        table_name = 'component'


class Dataset(Component):
    category = CharField(default='Dataset')
    snapshot = ForeignKeyField(Snapshot, backref='datasets')
    class Meta:
        table_name = 'component'


class Dataflow(Component):
    category = CharField(default='Dataflow')
    snapshot = ForeignKeyField(Snapshot, backref='dataflows')
    class Meta:
        table_name = 'component'


class Optimizer(Component):
    category = CharField(default='Optimizer')
    snapshot = ForeignKeyField(Snapshot, backref='optimizers')
    class Meta:
        table_name = 'component'


class Loss(Component):
    category = CharField(default='Loss')
    snapshot = ForeignKeyField(Snapshot, backref='losses')
    class Meta:
        table_name = 'component'


__all__ = ['Base', 'Experiment', 'Component', 'Model', 'Dataset', 'Dataflow', 'Optimizer', 'Loss']
