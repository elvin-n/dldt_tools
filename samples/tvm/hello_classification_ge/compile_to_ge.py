import argparse
import tvm
from tvm import relay
import onnx
from tvm.target import Target


parser = argparse.ArgumentParser(description=
    "tunes network on ios")
required = parser.add_argument_group('required arguments')
required.add_argument('-m', '--input_model', required=True, type=str, help="path to compiled .so file")
required.add_argument('-p', '--precision', required=False, type=str, help="precision to tune")
required.add_argument('-t', '--target', required=True, type=str, help="target for compilation")
required.add_argument('-l', '--host', required=True, type=str, help="host for linking")
args = parser.parse_args()

# verified with mobilenet v2 taken from onnx model zoo:
# https://media.githubusercontent.com/media/onnx/models/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx

def create_lib():
    name = args.input_model
    onnx_model = onnx.load(name)
    shape_dict = {}
    # There is bug in TVM on moment of this script creation, some stuff should be commented in
    # dense for GPU to make dynamic shapes to work
    # shape_dict["input"] = [relay.Any(), 3, 224, 224]
    shape_dict["input"] = [1, 3, 224, 224]
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    target = Target(args.target, host=args.host)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(model, target=target, params=params) 
    lib.export_library(f"{name}.ge.so")

if __name__ == '__main__':
    create_lib()


