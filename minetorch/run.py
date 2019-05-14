import minetorch
import torch
from minetorch import g, utils
import json

minetorch.core.boot()

def init_component(component_type, component_config):
    component_name = component_config['name']
    component_settings = component_config['settings']
    if component_type[-1] == 's':
        plural_component_cls_name = f'{component_type}es'
    else:
        plural_component_cls_name = f'{component_type}s'
    register_components = getattr(getattr(minetorch, component_type), f'registed_{plural_component_cls_name}')
    component = next((component for component in register_components if component.name == component_name), None)
    return component.func(**component_settings)

if __name__ == '__main__':
    with open('./config.json', 'r') as f:
        config = json.loads(f.read())

    g.dataset = init_component('dataset', config['dataset'])
    g.dataloader = torch.utils.data.DataLoader(g.dataset, batch_size=256, shuffle=True)
    # dataflow = init_component('dataflow', config['dataflow'])
    g.model = init_component('model', config['model'])
    print(config['model'])
    g.optimizer = init_component('optimizer', config['optimizer'])
    g.loss = init_component('loss', config['loss'])

    trainer = minetorch.Trainer(
        alchemistic_directory='./log',
        model=g.model,
        optimizer=g.optimizer,
        train_dataloader=g.dataloader,
        loss_func=g.loss
    )

    trainer.train()
