import onnxruntime as ort

def get_onnx_providers():
    # 获取可用的provider列表
    available_providers = ort.get_available_providers()
    print("Available ONNX Runtime providers:")
    for provider in available_providers:
        print(provider)

if __name__ == "__main__":
    get_onnx_providers()
