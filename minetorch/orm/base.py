from peewee import SqliteDatabase, Model, DateTimeField, TimestampField
from playhouse.shortcuts import model_to_dict
import datetime

db = SqliteDatabase('minetorch.db')

class Base(Model):
    created_at = DateTimeField(default=datetime.datetime.now)
    updated_at = TimestampField()

    def to_json_serializable(self):
        return model_to_dict(self)

    class Meta:
        database = db
        legacy_table_names = False