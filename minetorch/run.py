import minetorch
import json
minetorch.core.boot()

def find_component(component_type, component_name):
    if component_type[-1] == 's':
        plural_component_cls_name = f'{component_type}es'
    else:
        plural_component_cls_name = f'{component_type}s'
    register_components = getattr(getattr(minetorch, component_type), f'registed_{plural_component_cls_name}')
    return next((component for component in register_components if component.name == component_name), None)

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(open('config.json', 'r'))

    dataset = find_component('dataset', config['dataset']['name'])
    dataflow = find_component('dataflow', config['dataflow']['name'])
    model = find_component('model', config['model']['name'])
    optimizer = find_component('optimizer', config['optimizer']['name'])
    loss = find_component('loss', config['loss']['name'])

    trainer = minetorch.Trainer(
        alchemistic_directory='./log',
        model=model,
        optimizer=optimizer,
        train_dataloader=dataset,
        loss_func=loss
    )
