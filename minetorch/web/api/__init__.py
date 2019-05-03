from flask import Blueprint, jsonify, request, abort
import peewee
from minetorch import model, dataset, dataflow, loss, optimizer
from minetorch.orm import Experiment
from flask import render_template


api = Blueprint('api', 'api', url_prefix='/api')

@api.route('/models', methods=['GET'])
def models():
    return jsonify(list(map(lambda m: m.to_json_serializable(), model.registed_models)))

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
        Experiment.select().order_by(Experiment.updated_at.desc())
    )))

@api.route('/experiments', methods=['POST'])
def create_experiment():
    name = request.values['name']
    if not name: abort(422)
    try:
        experiment = Experiment.create(name=name)
    except peewee.IntegrityError: abort(409)
    return jsonify(experiment.to_json_serializable())

@api.errorhandler(422)
def entity_not_processable(error):
    return jsonify({'message': 'Entity is not processable'}), 422

@api.errorhandler(409)
def resource_conflict(error):
    return jsonify({'message': 'Resource already exists'}), 409
