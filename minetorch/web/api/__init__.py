from flask import Blueprint, jsonify

api = Blueprint('api', 'api', url_prefix='/api')

@api.route('/models', methods=['GET', 'POST'])
def models():
    return jsonify({'wtf': 'wtf'})
