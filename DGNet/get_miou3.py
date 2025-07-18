import os

from PIL import Image
from tqdm import tqdm

from dgnet import Unet
from utils.utils_metrics import compute_mIoU, show_results
import os
import time
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import GPUtil  # 需要 pip install gputil

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照JPG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
3、仅有按照VOC格式数据训练的模型可以利用这个文件进行miou的计算。
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = 3
    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    name_classes    = ["background","leaf", "gland"]
    # name_classes    = ["_background_","cat","dog"]
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'

    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        unet = Unet(num_classes=3, input_shape=[256, 256], model_path='logs/best_epoch_weights.pth')
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            image       = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)


    def measure_image_infer_time(image_list, repeat=3):
        times = []
        for _ in range(repeat):
            total_time = 0
            for img_name in image_list:
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path).convert('RGB')
                start = time.time()
                _ = unet.detect_image(image)  # 推理
                torch.cuda.synchronize()  # 强制同步，获取真实推理时间
                end = time.time()
                total_time += (end - start)
            times.append(total_time / len(image_list) * 1000)  # 转换为毫秒
        return np.array(times)

    print("🚀 正在批量处理图像并测量推理稳定性...")

    img_names = [f for f in os.listdir(dir_origin_path)
                if f.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))]

    if not os.path.exists(dir_save_path):
        os.makedirs(dir_save_path)

    # GPU预热
    print("⏱️ 正在进行 GPU warm-up...")
    for i in range(min(5, len(img_names))):
        image = Image.open(os.path.join(dir_origin_path, img_names[i])).convert('RGB')
        _ = unet.detect_image(image)
        torch.cuda.synchronize()

    # 测试推理时间稳定性
    max_rounds = 10
    threshold_std = 0.3  # ms
    print(f"📊 开始推理时间稳定性测试，最多 {max_rounds} 轮，每轮重复 3 次...")

    all_times = []
    for r in range(max_rounds):
        round_times = measure_image_infer_time(img_names, repeat=3)
        avg = round_times.mean()
        std = round_times.std()
        all_times.append(avg)
        print(f"第 {r+1} 轮：平均推理时间 = {avg:.2f} ms，标准差 = {std:.3f} ms")

        if r >= 2:
            recent_3 = all_times[-3:]
            if np.std(recent_3) < threshold_std:
                print("✅ 推理时间稳定，测试结束。")
                break

    final_avg_time_ms = np.mean(all_times[-3:])
    fps_val = 1000 / final_avg_time_ms

    # 保存预测结果图像
    # print("💾 正在保存预测结果图像...")
    # for img_name in tqdm(img_names):
    #     image_path = os.path.join(dir_origin_path, img_name)
    #     image = Image.open(image_path).convert('RGB')
    #     r_image = unet.detect_image(image)
    #     r_image.save(os.path.join(dir_save_path, img_name))

    # 显存使用情况（基于 GPUtil）
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        used_mem = gpu.memoryUsed
        total_mem = gpu.memoryTotal
        usage_ratio = used_mem / total_mem * 100
    else:
        used_mem = total_mem = usage_ratio = 0

    # 输出统计信息
    print("\n📌 性能统计结果：")
    print(f"平均推理时间（稳定）: {final_avg_time_ms:.2f} ms")
    print(f"FPS: {fps_val:.2f}")
    print(f"显存使用（峰值）：{used_mem:.2f} MB")
    print(f"GPU总显存：{total_mem:.2f} MB")
    print(f"显存占比：{usage_ratio:.2f}%")

    # 写入 txt 文件
    result_text = (
        f"模型批量预测性能统计结果（dir_predict 模式）\n"
        f"处理图像数：{len(img_names)} 张\n"
        f"平均推理时间（稳定）：{final_avg_time_ms:.2f} 毫秒\n"
        f"FPS：{fps_val:.2f} 帧/秒\n"
        f"显存使用（峰值）：{used_mem:.2f} MB\n"
        f"GPU 总显存：{total_mem:.2f} MB\n"
        f"显存占比：{usage_ratio:.2f}%\n"
    )

    with open("fps_results.txt", "w", encoding="utf-8") as f:
        f.write(result_text)

    print("\n✅ 推理统计结果已写入 fps_results.txt")
