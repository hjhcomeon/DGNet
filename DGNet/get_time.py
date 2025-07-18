import torch
import torch.nn as nn
import time
from torchinfo import summary
from nets.DGNet import get_seg_model  # 导入你的模型

def measure_fps(model, input_shape, device, test_interval=100):
    """
    测量模型的FPS（每秒帧数）。
    :param model: 模型
    :param input_shape: 输入图像的大小 (H, W)
    :param device: 设备 (CPU 或 GPU)
    :param test_interval: 测试间隔，默认为100次推理
    :return: FPS
    """
    model.eval()
    model.to(device)
    
    # 创建一个虚拟输入
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    
    # 预热模型
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)
    
    # 测试推理时间
    start_time = time.time()
    with torch.no_grad():
        for _ in range(test_interval):
            model(dummy_input)
    end_time = time.time()
    
    fps = test_interval / (end_time - start_time)
    return fps

def measure_flops_params(model, input_shape):
    """
    测量模型的FLOPs和参数量。
    :param model: 模型
    :param input_shape: 输入图像的大小 (H, W)
    :return: FLOPs 和 Params
    """
    device = torch.device("cpu")  # 使用CPU进行FLOPs计算
    model.eval()
    model.to(device)
    
    summary_result = summary(model, input_size=(1, 3, input_shape[0], input_shape[1]), device=device)
    
    flops = summary_result.total_mult_adds / 1e9  # 转换为GFLOPs
    params = summary_result.total_params / 1e6  # 转换为MParams
    
    return flops, params

if __name__ == "__main__":
    # 模型参数
    num_classes = 3
    input_shape = [256, 256]  # 输入图像大小
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    # model = get_seg_model(input_shape + [3], num_classes)  # 修改为你的模型加载方式
    model = get_seg_model(num_classes)  # 修改为你的模型加载方式
    model.to(device)
    
    # 测量FPS
    fps = measure_fps(model, input_shape, device)
    print(f"FPS: {fps:.2f}")
    
    # 测量FLOPs和Params
    flops, params = measure_flops_params(model, input_shape)
    print(f"FLOPs: {flops:.2f} GFLOPs")
    print(f"Params: {params:.2f} MParams")
