from flask import Blueprint, jsonify
from minetorch import model, dataset, dataflow, loss, optimizer

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
