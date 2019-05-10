from flask import Blueprint, jsonify, request, abort, g
import peewee
from minetorch import model, dataset, dataflow, loss, optimizer
from minetorch.orm import Experiment, Model, Snapshot, Dataset, Dataflow, Optimizer, Loss
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

def create_component(component_class):
    name = request.values['name']
    if not name: abort(422)
    component = component_class.create(
        name=name,
        settings=request.values.get('settings'),
        snapshot_id=g.snapshot.id
    )
    return jsonify(component.to_json_serializable())

def get_component(component_class, component_id):
    try:
        component = component_class.get_by_id(component_id)
    except peewee.DoesNotExist: abort(404)
    return jsonify(component.to_json_serializable())

@experiment.route('/datasets', methods=['POST'])
def create_dataset(experiment_id):
    return create_component(Dataset)

@experiment.route('/datasets/<dataset_id>', methods=['GET'])
def get_dataset(experiment_id, dataset_id):
    return get_component(Dataset, dataset_id)

@experiment.route('/datasets', methods=['GET'])
def datasets_list(experiment_id):
    return jsonify(list(map(lambda m: m.to_json_serializable(), dataset.registed_datasets)))

@experiment.route('/dataflows', methods=['POST'])
def create_dataflow(experiment_id):
    return create_component(Dataflow)

@experiment.route('/dataflows/<dataflow_id>', methods=['GET'])
def get_dataflow(experiment_id, dataflow_id):
    return get_component(Dataflow, dataflow_id)

@experiment.route('/dataflows', methods=['GET'])
def dataflows_list(experiment_id):
    return jsonify(list(map(lambda m: m.to_json_serializable(), dataflow.registed_dataflows)))

@experiment.route('/optimizers', methods=['POST'])
def create_optimizer(experiment_id):
    return create_component(Optimizer)

@experiment.route('/optimizers/<optimizer_id>', methods=['GET'])
def get_optimizer(experiment_id, optimizer_id):
    return get_component(Optimizer, optimizer_id)

@experiment.route('/optimizers', methods=['GET'])
def optimizers_list(experiment_id):
    return jsonify(list(map(lambda m: m.to_json_serializable(), dataset.registed_optimizers)))

@experiment.route('/losses', methods=['POST'])
def create_loss(experiment_id):
    return create_component(Loss)

@experiment.route('/losses/<loss_id>', methods=['GET'])
def get_loss(loss_id):
    return get_component(Loss, loss_id)

@experiment.route('/losses', methods=['GET'])
def losses_list(experiment_id):
    return jsonify(list(map(lambda m: m.to_json_serializable(), loss.registed_losses)))

@experiment.route('/models', methods=['POST'])
def create_model(experiment_id):
    return create_component(Model)

@experiment.route('/model/<model_id>', methods=['GET'])
def get_model(experiment_id, model_id):
    return get_component(Model, model_id)

@experiment.route('/models', methods=['GET'])
def models_list(experiment_id):
    return jsonify(list(map(lambda m: m.to_json_serializable(), model.registed_models)))

@api.errorhandler(422)
def entity_not_processable(error):
    return jsonify({'message': 'Entity is not processable'}), 422

@api.errorhandler(409)
def resource_conflict(error):
    return jsonify({'message': 'Resource already exists'}), 409
