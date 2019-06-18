from minetorch.orm import Component, Experiment, Snapshot,
                          Timer, Graph, Point, Workflow
from torchvision.datasets.mnist import MNIST


def create_sample_experiment(experiment_name):
    """Create a sample experiment for easy testing and develop.
    The sample experiment enabled the Minetorch default package
    and has been configured with 3 workflows:
      - Training Workflow with Minetorch/CoreTrainingPlugin
        * `Dataset` Minetorch/MNIST-training
        * `Dataflow` Minetorch/Resize(28)
        * `Dataflow` Minetorch/RandomHorizontalFlip(0.5)
        * `Dataflow` Minetorch/Tensorize(norm=True)
        * `Model` MinetorchDemo/LeNet
        * `Loss` Minetorch/CrossEntropy
        * `Optimizer` Minetorch/SGD
      - Validation Workflow Minetorch/CoreValidationPlugin
        * `Dataset` Minetorch/MNIST-validation
        * `Dataflow` Minetorch/Resize(28)
        * `Dataflow` Minetorch/Tensorize(norm=True)
        * `Model` MinetorchDemo/LeNet
        * `Custom` MinetorchDemo/ComputeAccuracy
        * `Custom` MinetorchDemo/ComputeF1
      - Test Workflow with Minetorch/CoreValidationPlugin
        * `Dataset` MinetorchDemo/HandWrittenDigits
        * `Dataflow` Minetorch/Resize(28)
        * `Dataflow` Minetorch/Tensorize(norm=True)
        * `Model` MinetorchDemo/LeNet
        * `Custom` MinetorchDemo/PredictHandWrittenDigits
    """
    # create experiment and snapshot
    experiment = Experiment.create(name=name)
    snapshot = experiment.create_draft_snapshot()
    snapshot.update(category=1).execute()

    # create workflows
    training_workflow = Workflow.create(
        name='Training Workflow',
        snapshot=snapshot,
        plugins=['Minetorch/CoreTrainingPlugin'],
    )
    validation_workflow = Workflow.create(
        name='Validation Workflow',
        snapshot=snapshot,
        plugins=['Minetorch/CoreValidationPlugin']
    )
    test_workflow = Workflow.create(
        name='Validation Workflow',
        snapshot=snapshot,
        plugins=['Minetorch/CoreValidationPlugin']
    )
    validation_workflow.set_prev_workflow(training_workflow)
    test_workflow.set_prev_workfow(validation_workflow)

    # split dataset

    # create training components
