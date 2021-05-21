import sys
import onnx
import json
from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse

json_path = sys.argv[1]
onnx_path = sys.argv[2]

# Convert onnx model to JSON
# model_path = onnx_path
# onnx_model = onnx.load(model_path)
# print(onnx_model)
# s = MessageToJson(onnx_model)
# onnx_json = json.loads(s)

# Convert JSON to String
json_path = sys.argv[1]
f = open(json_path,)
onnx_json = json.load(f)
onnx_str = json.dumps(onnx_json)

# Convert String to onnx model
convert_model = Parse(onnx_str, onnx.ModelProto())

onnx.save(convert_model, onnx_path)
