import datetime
import json
import logging
import functools

import peewee
from peewee import (CharField, DateTimeField, Field, ForeignKeyField,
                    IntegerField, FloatField)
from peewee import Model as PeeweeModel
from peewee import SqliteDatabase, TextField
from playhouse.shortcuts import model_to_dict
from minetorch.utils import server_file

logger = logging.getLogger('peewee')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

db_file = server_file('minetorch.db')
db = SqliteDatabase(db_file)


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
    deleted_at = DateTimeField(null=True)

    def to_json_serializable(self):
        return model_to_dict(self)

    class Meta:
        database = db
        legacy_table_names = False


class Experiment(Base):
    name = CharField(unique=True)
    total_training_time = IntegerField(default=0)
    # 1: stopped 2: idle 3: training
    status = IntegerField(default=1, null=True)
    last_stopped_at = DateTimeField(null=True)
    last_started_at = DateTimeField(null=True)

    def create_draft_snapshot(self):
        """Try to create a draft snapshot, if there's already a draft version,
        should raise an error, otherwise it should check if there already have an
        current version, if yes, then clone it and return, if no, create a brand new
        snapshot(this is offen right after the Experiment is been created)
        """
        draft_snapshot = self.draft_snapshot()
        if draft_snapshot:
            return draft_snapshot
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

    def publish(self):
        draft = self.draft_snapshot()
        current = self.current_snapshot()
        if not draft:
            draft = self.create_draft_snapshot()
        draft.category = 1
        draft.save()

        if current:
            current.category = 2
            current.save()

        self.save()


class Snapshot(Base):
    total_training_time = IntegerField(default=0)
    stopped_at = DateTimeField(null=True)
    # 0: draft 1: current 2: archived
    category = IntegerField(default=0)
    code = TextField(null=True)
    experiment = ForeignKeyField(Experiment, backref='snapshots')

    def clone(self):
        new_snapshot = Snapshot.create(experiment=self.experiment)
        for component_name in ['datasets', 'dataflows', 'models', 'optimizers', 'losses']:
            components = getattr(self, component_name)
            if len(components) == 0:
                continue
            components[0].clone(new_snapshot)
        # force reload
        return Snapshot.get_by_id(new_snapshot.id)

    def is_draft(self):
        return self.category == 0

    def is_current(self):
        return self.category == 1

    def is_archived(self):
        return self.category == 2


class Workflow(Base):
    """Workflow is a container of ordered Components. Workflow can be chained
    to do more complicated stuff. For instance, by most cases, there will be 2
    workflow in order: train workflow -> val workflow
    """
    name = CharField()
    next_workflow_id = IntegerField(null=True)
    snapshot = ForeignKeyField(Snapshot, backref='components')
    plugins = CharField(null=True)

    class Meta:
        indexes = ('next_workflow_id')

    def set_prev_workflow(self, prev_workflow_id):
        try:
            prev_workflow = Workflow.get_by_id(prev_workflow_id)
        except peewee.DoesNotExist:
            return None
        prev_workflow.update(next_workflow_id=self.id).execute()
        return prev_workflow


class Component(Base):
    name = CharField()
    category = CharField()
    settings = JsonField(null=True)
    snapshot = ForeignKeyField(Snapshot, backref='components')
    workflow = ForeignKeyField(Workflow, backref='components')
    next_component_id = IntegerField(null=True)
    code = TextField(null=True)

    class Meta:
        indexes = (
            (('name', 'snapshot'), True),
            'next_component_id'
        )

    def clone(self, snapshot):
        return Component.create(
            snapshot=snapshot,
            name=self.name,
            category=self.category,
            settings=self.settings,
            code=self.code
        )

    @classmethod
    def select(cls):
        return super().select().where(cls.category == str(cls.__name__))

    @classmethod
    def create(cls, **query):
        if 'category' not in query:
            query['category'] = cls.__name__
        return super().create(**query)

    def set_prev_component(self, prev_component_id):
        try:
            prev_component = Component.get_by_id(prev_component_id)
        except peewee.DoesNotExist:
            return None
        prev_component.update(next_component_id=self.id).execute()
        return prev_component


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


class Timer(Base):
    experiment = ForeignKeyField(Experiment, backref='timers')
    snapshot = ForeignKeyField(Snapshot, backref='timers')
    name = CharField(null=True)
    # 1. iteration 2. epoch 3. snapshot
    category = CharField()
    current = IntegerField()
    ratio = IntegerField(null=True)


class Graph(Base):
    experiment = ForeignKeyField(Experiment, backref='graphs')
    snapshot = ForeignKeyField(Snapshot, backref='graphs')
    timer = ForeignKeyField(Timer, backref='graphs')
    name = CharField()
    sequence = FloatField(null=True)
    key = CharField(null=True)

    def add_point(self, x, y):
        Point.create(graph=self, x=x, y=y)

    class Meta:
        indexes = (
            (('experiment_id', 'name'), True),
        )


class Point(PeeweeModel):
    """For performance consideration, this will not inherit from the Base class
    """
    graph = ForeignKeyField(Graph, backref='points')
    x = IntegerField()
    y = CharField()

    class Meta:
        database = db
        legacy_table_names = False

    def to_json_serializable(self):
        return {
            'x': self.x,
            'y': self.y
        }


def reduce_callback(mapping, model):
    mapping[model.__name__] = model
    return mapping


components_mapping = functools.reduce(reduce_callback, Component.__subclasses__(), {})


def find_component(name):
    return components_mapping.get(name, None)


__all__ = ['Base', 'Experiment', 'Component', 'Model', 'Dataset', 'Dataflow',
           'Optimizer', 'Loss', 'Graph', 'Point', 'Timer', 'Workflow', 'find_component']
