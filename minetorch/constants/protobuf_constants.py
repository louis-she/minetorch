from minetorch.rpc.grpc import minetorch_pb2


COMMAND_TRAIN = minetorch_pb2.YoMessage.Command.Value('TRAIN')
COMMAND_HALT = minetorch_pb2.YoMessage.Command.Value('HALT')
COMMAND_KILL = minetorch_pb2.YoMessage.Command.Value('KILL')

STATUS_TRAINING = minetorch_pb2.HeyMessage.Status.Value('TRAINING')
STATUS_IDLE = minetorch_pb2.HeyMessage.Status.Value('IDLE')
STATUS_ERROR = minetorch_pb2.HeyMessage.Status.Value('ERROR')

TIMER_ITERATION = minetorch_pb2.Timer.Category.Value('ITERATION')
TIMER_EPOCH = minetorch_pb2.Timer.Category.Value('EPOCH')
TIMER_SNAPSHOT = minetorch_pb2.Timer.Category.Value('SNAPSHOT')
