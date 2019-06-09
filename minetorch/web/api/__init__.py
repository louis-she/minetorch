import json
import logging
from pathlib import Path

import peewee
from flask import Blueprint, abort, g, jsonify, request
from minetorch import dataflow, dataset, loss, model, optimizer
from minetorch.core import setup_runtime_directory
from minetorch.orm import (Dataflow, Dataset, Experiment, Loss,
                           Model, Optimizer, Graph)
from multiprocessing import Process
from minetorch.runtime import main_process
from minetorch.utils import runtime_file

api = Blueprint('api', 'api', url_prefix='/api')
experiment = Blueprint('experiment', 'experiment', url_prefix='/api/experiments/<experiment_id>')
graph = Blueprint('graph', 'graph', url_prefix='/api/graphs/<graph_id>')


@experiment.before_request
def experiment_before_request():
    experiment_id = request.view_args['experiment_id']
    g.experiment = Experiment.get(id=experiment_id)
    g.snapshot = g.experiment.draft_snapshot()
    if not g.snapshot:
        g.snapshot = g.experiment.create_draft_snapshot()


@graph.before_request
def graph_before_request():
    experiment_id = request.view_args['graph_id']
    g.graph = Graph.get(id=experiment_id)


@experiment.route('', methods=['DELETE'])
def delete_experiment(experiment_id):
    g.experiment.delete()
    return jsonify({'message': 'ok'})


@experiment.route('/running', methods=['POST'])
def train_experiment():
    pass


@api.route('/dataflows', methods=['GET'])
def dtaflows():
    return jsonify(list(map(lambda m: m.to_json_serializable(), dataflow.registed_dataflows)))


@api.route('/losses', methods=['GET'])
def losses():
    return jsonify(list(map(lambda m: m.to_json_serializable(), loss.registed_losses)))


@api.route('/optimizers', methods=['GET'])
def optimizers():
    return jsonify(list(map(lambda m: m.to_json_serializable(), optimizer.registed_optimizers)))


@api.route('/experiments', methods=['GET'])
def experiments_list():
    return jsonify(list(map(
        lambda m: m.to_json_serializable(),
        Experiment.select().where(Experiment.deleted_at.is_null()).order_by(Experiment.created_at.desc())
    )))


@experiment.route('', methods=['GET'])
def get_experiment(experiment_id):
    experiment = g.experiment.to_json_serializable()
    experiment['snapshot'] = g.experiment.current_snapshot().to_json_serializable()
    for component in [Dataset, Dataflow, Model, Optimizer, Loss]:
        experiment[component.__name__.lower()] = get_component(component, False)
    return jsonify(experiment)


@api.route('/experiments', methods=['POST'])
def create_experiment():
    name = request.values['name']
    if not name:
        abort(422)
    try:
        experiment = Experiment.create(name=name)
    except peewee.IntegrityError:
        abort(409)
    experiment.create_draft_snapshot()
    return jsonify(experiment.to_json_serializable())


def create_component(component_class):
    name = request.values['name']
    if not name:
        abort(422)

    settings = request.values.to_dict()
    settings.pop('name')
    try:
        component = component_class.create(
            name=name,
            settings=json.dumps(settings),
            snapshot_id=g.snapshot.id
        )
    except peewee.IntegrityError:
        abort(409)

    component_class.delete().where(
        (component_class.snapshot == g.snapshot) &
        (component_class.id != component.id) &
        (component_class.category == component.category)
    ).execute()
    return jsonify(component.to_json_serializable())


def get_component(component_class, should_jsonify=True):
    try:
        component = component_class.get(component_class.snapshot == g.snapshot)
    except peewee.DoesNotExist:
        return jsonify({})
    component_dict = component.to_json_serializable()
    if should_jsonify:
        return jsonify(component_dict)
    return component_dict


def update_component(component_class):
    try:
        component = component_class.select().where(component_class.snapshot == g.snapshot).get()
    except peewee.DoesNotExist:
        return abort(404)
    component.settings = json.dumps(request.values.to_dict())
    component.save()
    return jsonify(component.to_json_serializable())


@experiment.route('/datasets', methods=['GET'])
def datasets_list(experiment_id):
    return jsonify(list(map(lambda m: m.to_json_serializable(), dataset.registed_datasets)))


@experiment.route('/datasets', methods=['POST'])
def create_dataset(experiment_id):
    return create_component(Dataset)


@experiment.route('/datasets/selected', methods=['GET'])
def get_dataset(experiment_id):
    return get_component(Dataset)


@experiment.route('/datasets/selected', methods=['PATCH'])
def update_dataset(experiment_id):
    return update_component(Dataset)


@experiment.route('/dataflows', methods=['GET'])
def dataflows_list(experiment_id):
    return jsonify(list(map(lambda m: m.to_json_serializable(), dataflow.registed_dataflows)))


@experiment.route('/dataflows', methods=['POST'])
def create_dataflow(experiment_id):
    return create_component(Dataflow)


@experiment.route('/dataflows/selected', methods=['GET'])
def get_dataflow(experiment_id):
    return get_component(Dataflow)


@experiment.route('/dataflows/selected', methods=['PATCH'])
def update_dataflow(experiment_id):
    return update_component(Dataflow)


@experiment.route('/optimizers', methods=['GET'])
def optimizers_list(experiment_id):
    return jsonify(list(map(lambda m: m.to_json_serializable(), optimizer.registed_optimizers)))


@experiment.route('/optimizers', methods=['POST'])
def create_optimizer(experiment_id):
    return create_component(Optimizer)


@experiment.route('/optimizers/selected', methods=['GET'])
def get_optimizer(experiment_id):
    return get_component(Optimizer)


@experiment.route('/optimizers/selected', methods=['PATCH'])
def update_optimizer(experiment_id):
    return update_component(Optimizer)


@experiment.route('/losses', methods=['GET'])
def losses_list(experiment_id):
    return jsonify(list(map(lambda m: m.to_json_serializable(), loss.registed_losses)))


@experiment.route('/losses', methods=['POST'])
def create_loss(experiment_id):
    return create_component(Loss)


@experiment.route('/losses/selected', methods=['GET'])
def get_loss(experiment_id):
    return get_component(Loss)


@experiment.route('/losses/selected', methods=['PATCH'])
def update_loss(experiment_id):
    return update_component(Loss)


@experiment.route('/models', methods=['POST'])
def create_model(experiment_id):
    return create_component(Model)


@experiment.route('/models/selected', methods=['GET'])
def get_model(experiment_id):
    return get_component(Model)


@experiment.route('/models', methods=['GET'])
def models_list(experiment_id):
    return jsonify(list(map(lambda m: m.to_json_serializable(), model.registed_models)))


@experiment.route('/models/selected', methods=['PATCH'])
def update_model(experiment_id):
    return update_component(Model)


@experiment.route('/publish', methods=['POST'])
def create_publish(experiment_id):
    g.experiment.publish()
    setup_runtime_directory(g.experiment)
    return jsonify({'message': 'ok'})


@experiment.route('/training', methods=['POST'])
def start_train(experiment_id):
    if g.experiment.status == 1:
        create_runtime_process(g.experiment)
    g.experiment.status = 3
    g.experiment.save()
    return jsonify({'message': 'ok'})


@experiment.route('/halt', methods=['POST'])
def halt_train(experiment_id):
    g.experiment.status = 2
    g.experiment.save()
    return jsonify({'message': 'ok'})

# Graphs
@experiment.route('/graphs', methods=['GET'])
def graphs(experiment_id):
    return jsonify(list(map(lambda m: m.to_json_serializable(), g.experiment.graphs)))


@graph.route('/points', methods=['GET'])
def points(graph_id):
    return jsonify(list(map(lambda m: m.to_json_serializable(), g.graph.points)))


@api.errorhandler(422)
def entity_not_processable(error):
    return jsonify({'message': 'Entity is not processable'}), 422


@api.errorhandler(409)
def resource_conflict(error):
    return jsonify({'message': 'Resource already exists'}), 409


def create_runtime_process(experiment):
    # Check if the experiment status is stopped. if yes, create the training process
    logging.info('creating runtime')
    runtime_process = Process(target=main_process, args=(runtime_file(Path(str(experiment.current_snapshot().id)) / 'config.json', experiment),))
    # TODO: maybe we should detach the child process
    # runtime_process.daemon = True
    runtime_process.start()
    logging.info('runtime created')
