from flask import Blueprint, jsonify, request, abort, g
import peewee
from minetorch import model, dataset, dataflow, loss, optimizer
from minetorch.orm import Experiment, Model, Snapshot, Dataset, Dataflow, Optimizer
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

@experiment.route('/dataset', methods=['POST'])
def create_dataset():
    """Create a dataset for a snapshot
    """
    name = request.values['name']
    if not name: abort(422)
    dataset = Dataset.create(
        name = name,
        settings = request.values['settings'],
        code = request.values['code'],
        snapshot_id=g.snapshot.id
    )
    return jsonify(dataset.to_json_serializable())

@experiment.route('/dataset/<dataset_id>', methods=['GET'])
def get_dataset():
    """Get a dataset
    """
    dataset_id = request.view_args['dataset_id']
    if not dataset_id: abort(422)
    try:
        dataset = Dataset.get_by_id(dataset_id)
    except peewee.DoesNotExist: abort(409)
    return jsonify(dataset.to_json_serializable())

@experiment.route('/datasets', mothods=['GET'])
def datasets_list():
    """Get all datasets
    """
    return jsonify(list(map(lambda m: m.to_json_serializable(), dataset.registed_datasets)))


@experiment.route('/dataflow', methods=['POST'])
def create_dataflow():
    """Create a dataflow for a snapshot
    """
    name = request.values['name']
    if not name: abort(422)
    dataflow = Dataflow.create(
        name = name,
        settings = request.values['settings'],
        code = request.values['code'],
        snapshot_id=g.snapshot.id
    )
    return jsonify(dataflow.to_json_serializable())

@experiment.route('/dataflow/<dataflow_id>', methods=['GET'])
def get_dataflow():
    """Get a dataflow
    """
    dataflow_id = request.view_args['dataflow_id']
    if not dataflow_id: abort(422)
    try:
        dataflow = Dataset.get_by_id(dataflow_id)
    except peewee.DoesNotExist: abort(409)
    return jsonify(dataflow.to_json_serializable())

@experiment.route('/dataflows', mothods=['GET'])
def dataflows_list():
    """Get all dataflows
    """
    return jsonify(list(map(lambda m: m.to_json_serializable(), dataflow.registed_dataflows)))


@experiment.route('/optimizer', methods=['POST'])
def create_optimizer():
    """Create a optimizer for a snapshot
    """
    name = request.values['name']
    if not name: abort(422)
    optimizer = Optimizer.create(
        name = name,
        settings = request.values['settings'],
        code = request.values['code'],
        snapshot_id=g.snapshot.id
    )
    return jsonify(optimizer.to_json_serializable())

@experiment.route('/optimizer/<optimizer_id>', methods=['GET'])
def get_optimizer():
    """Get a optimizer
    """
    optimizer_id = request.view_args['optimizer_id']
    if not optimizer_id: abort(422)
    try:
        optimizer = Optimizer.get_by_id(optimizer_id)
    except peewee.DoesNotExist: abort(409)
    return jsonify(optimizer.to_json_serializable())

@experiment.route('/optimizers', mothods=['GET'])
def optimizers_list():
    """Get all optimizers
    """
    return jsonify(list(map(lambda m: m.to_json_serializable(), dataset.registed_optimizers)))

@experiment.route('/loss', methods=['POST'])
def create_loss():
    """Create a loss for a snapshot
    """
    name = request.values['name']
    if not name: abort(422)
    loss = Loss.create(
        name = name,
        settings = request.values['settings'],
        code = request.values['code'],
        snapshot_id=g.snapshot.id
    )
    return jsonify(loss.to_json_serializable())

@experiment.route('/loss/<loss_id>', methods=['GET'])
def get_loss():
    """Get a loss
    """
    loss_id = request.view_args['loss_id']
    if not loss_id: abort(422)
    try:
        loss = Loss.get_by_id(loss_id)
    except peewee.DoesNotExist: abort(409)
    return jsonify(loss.to_json_serializable())

@experiment.route('/losses', mothods=['GET'])
def losses_list():
    """Get all losses
    """
    return jsonify(list(map(lambda m: m.to_json_serializable(), loss.registed_losses)))

@experiment.route('/model', methods=['POST'])
def create_model():
    """Create a model for a snapshot
    """
    name = request.values['name']
    if not name: abort(422)
    model = Model.create(
        name = name,
        settings = request.values['settings'],
        code = request.values['code'],
        snapshot_id=g.snapshot.id
    )
    return jsonify(model.to_json_serializable())

@experiment.route('/model/<model_id>', methods=['GET'])
def get_model():
    """Get a model
    """
    model_id = request.view_args['model_id']
    if not model_id: abort(422)
    try:
        model = Model.get_by_id(model_id)
    except peewee.DoesNotExist: abort(409)
    return jsonify(model.to_json_serializable())

@experiment.route('/models', mothods=['GET'])
def models_list():
    """Get all models
    """
    return jsonify(list(map(lambda m: m.to_json_serializable(), model.registed_models)))




@api.errorhandler(422)
def entity_not_processable(error):
    return jsonify({'message': 'Entity is not processable'}), 422

@api.errorhandler(409)
def resource_conflict(error):
    return jsonify({'message': 'Resource already exists'}), 409
