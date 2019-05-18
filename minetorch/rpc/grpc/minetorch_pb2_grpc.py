# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from minetorch.rpc.grpc import minetorch_pb2 as minetorch_dot_rpc_dot_grpc_dot_minetorch__pb2


class MinetorchStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.HeyYo = channel.unary_unary(
        '/minetorch.Minetorch/HeyYo',
        request_serializer=minetorch_dot_rpc_dot_grpc_dot_minetorch__pb2.HeyMessage.SerializeToString,
        response_deserializer=minetorch_dot_rpc_dot_grpc_dot_minetorch__pb2.YoMessage.FromString,
        )


class MinetorchServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def HeyYo(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_MinetorchServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'HeyYo': grpc.unary_unary_rpc_method_handler(
          servicer.HeyYo,
          request_deserializer=minetorch_dot_rpc_dot_grpc_dot_minetorch__pb2.HeyMessage.FromString,
          response_serializer=minetorch_dot_rpc_dot_grpc_dot_minetorch__pb2.YoMessage.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'minetorch.Minetorch', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))