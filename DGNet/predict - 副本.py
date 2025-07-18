#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from unet import  Unet




if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    #-------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #   'predict_onnx'      表示利用导出的onnx模型进行预测，相关参数的修改在unet.py_346行左右处的Unet_ONNX
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    #-------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    #
    #   count、name_classes仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    count           = False
    name_classes    = ["background", "gland","leaf"]
    # name_classes    = ["background","cat","dog"]
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode != "predict_onnx":
        unet = Unet()
    #else:
        #yolo = Unet_ONNX()

    if mode == "predict":
        '''
        predict.py有几个注意点
        1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
        具体流程可以参考get_miou_prediction.py，在get_miou_prediction.py即实现了遍历。
        2、如果想要保存，利用r_image.save("img.jpg")即可保存。
        3、如果想要原图和分割图不混合，可以把blend参数设置成False。
        4、如果想根据mask获取对应的区域，可以参考detect_image函数中，利用预测结果绘图的部分，判断每一个像素点的种类，然后根据种类获取对应的部分。
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = unet.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)  # BGR转RGB
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(unet.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)  # RGB转BGR
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = unet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')


        
    #elif mode == "dir_predict":
        #import os
        #from tqdm import tqdm

        #img_names = os.listdir(dir_origin_path)
        #for img_name in tqdm(img_names):
            #if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                #image_path  = os.path.join(dir_origin_path, img_name)
                #image       = Image.open(image_path)
                #r_image     = unet.detect_image(image)
                #if not os.path.exists(dir_save_path):
                    #os.makedirs(dir_save_path)
                #r_image.save(os.path.join(dir_save_path, img_name))
    
    elif mode == "dir_predict":
        import os
        import numpy as np
        from tqdm import tqdm
        from time import time
        from PIL import Image
        import tensorflow as tf

        def measure_image_infer_time(image_list, repeat=3):
            times = []
            for _ in range(repeat):
                total_time = 0
                for img_name in image_list:
                    image_path = os.path.join(dir_origin_path, img_name)
                    image = Image.open(image_path)
                    start = time()
                    _ = unet.detect_image(image)
                    # TensorFlow 默认同步，没必要显式同步，除非你用了 tf.function
                    end = time()
                    total_time += (end - start)
                times.append(total_time / len(image_list) * 1000)  # ms
            return np.array(times)

        print("🚀 正在批量处理图像并测量推理稳定性...")

        img_names = [f for f in os.listdir(dir_origin_path)
                    if f.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))]

        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)

        # GPU预热：连续推理几张图片
        print("⏱️ 正在进行 GPU warm-up...")
        for i in range(min(5, len(img_names))):
            image = Image.open(os.path.join(dir_origin_path, img_names[i]))
            _ = unet.detect_image(image)

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
        print("💾 正在保存预测结果图像...")
        for img_name in tqdm(img_names):
            image_path = os.path.join(dir_origin_path, img_name)
            image = Image.open(image_path)
            r_image = unet.detect_image(image)
            r_image.save(os.path.join(dir_save_path, img_name))

        # 获取显存使用情况（需要TF2.3+）
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                print(f"GPU显存使用情况: {gpu.memoryUsed} MB / {gpu.memoryTotal} MB ({gpu.memoryUtil*100:.2f}%)")
            else:
                print("未检测到GPU信息")

            #used_mem = mem_info['peak'] / (1024 * 1024)  # MB
            # GPU总显存没直接API，需要你预先知道或手动设置
            # 这里给个示例，手动设置总显存，比如16GB：
            total_mem = 16 * 1024  # MB，根据你的GPU实际显存修改
            usage_ratio = used_mem / total_mem * 100
        else:
            used_mem = 0
            total_mem = 0
            usage_ratio = 0

        print("\n📌 性能统计结果：")
        print(f"平均推理时间（稳定）: {final_avg_time_ms:.2f} ms")
        print(f"FPS: {fps_val:.2f}")
        print(f"显存使用（峰值）：{used_mem:.2f} MB")
        print(f"GPU总显存（手动设置）：{total_mem:.2f} MB")
        print(f"显存占比：{usage_ratio:.2f}%")

        # 写入 txt 文件
        result_text = (
            f"模型批量预测性能统计结果（dir_predict 模式）\n"
            f"处理图像数：{len(img_names)} 张\n"
            f"平均推理时间（稳定）：{final_avg_time_ms:.2f} 毫秒\n"
            f"FPS：{fps_val:.2f} 帧/秒\n"
            f"显存使用（峰值）：{used_mem:.2f} MB\n"
            f"GPU 总显存（手动设置）：{total_mem:.2f} MB\n"
            f"显存占比：{usage_ratio:.2f}%\n"
        )

        with open("fps_results.txt", "w", encoding="utf-8") as f:
            f.write(result_text)

        print("\n✅ 推理统计结果已写入 fps_results.txt")


   

    elif mode == "export_onnx":
        unet.convert_to_onnx(simplify, onnx_save_path)
