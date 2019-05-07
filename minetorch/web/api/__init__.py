from flask import Blueprint, jsonify, request, abort, g
import peewee
from minetorch import model, dataset, dataflow, loss, optimizer
from minetorch.orm import Experiment, Model, Snapshot
from flask import render_template

api = Blueprint('api', 'api', url_prefix='/api')
experiment = Blueprint('experiment', 'experiment', url_prefix='/api/experiments/<experiment_id>')

@experiment.before_request
def experiment_before_request():
    experiment_id = request.view_args['experiment_id']
    g.experiment = Experiment.get(id=experiment_id)
    g.snapshot = g.experiment.draft_snapshot()
    if not g.snapshot:
        g.snapshot = g.experiment.create_draft_snapshot()

@api.route('/models', methods=['GET'])
def models():
    """List all the available models
    """
    return jsonify(list(map(lambda m: m.to_json_serializable(), model.registed_models)))

@experiment.route('', methods=['DELETE'])
def delete_experiment(experiment_id):
    g.experiment.delete()
    return jsonify({'message': 'ok'})

@experiment.route('/running', methods=['POST'])
def train_experiment():
    pass

@experiment.route('/models', methods=['POST'])
def creat_model(experiment_id):
    """Pick a model for an experiment
    """
    model = Model.create(
        name=request.values['name'],
        settings=request.values['settings'],
        snapshot_id=g.snapshot.id
    )
    return jsonify(model.to_json_serializable())

@api.route('/datasets', methods=['GET'])
def datasets():
    return jsonify(list(map(lambda m: m.to_json_serializable(), dataset.registed_datasets)))

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
        Experiment.select().where(Experiment.deleted_at == None).order_by(Experiment.updated_at.desc())
    )))

@api.route('/experiments', methods=['POST'])
def create_experiment():
    name = request.values['name']
    if not name: abort(422)
    try:
        experiment = Experiment.create(name=name)
    except peewee.IntegrityError: abort(409)
    experiment.create_draft_snapshot()
    return jsonify(experiment.to_json_serializable())

@api.errorhandler(422)
def entity_not_processable(error):
    return jsonify({'message': 'Entity is not processable'}), 422

@api.errorhandler(409)
def resource_conflict(error):
    return jsonify({'message': 'Resource already exists'}), 409
