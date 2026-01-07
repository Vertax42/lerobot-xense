import onnxruntime
from onnxruntime import set_default_logger_severity
set_default_logger_severity(1)

session = onnxruntime.InferenceSession("./test.onnx", providers=[ 'CUDAExecutionProvider', 'CPUExecutionProvider'])

if "CUDAExecutionProvider" in session.get_providers():
    print("infer session using GPU")
else:
    print("infer session using CPU")