#!/usr/bin/env python3
import sys

try:
    import pynvml
except ImportError:
    sys.exit("请先安装 pynvml: pip install nvidia-ml-py3")

def check_gpu_ecc():
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        sys.exit("NVML 初始化失败: " + str(e))
        
    device_count = pynvml.nvmlDeviceGetCount()
    bad_gpus = []
    print("检测到 GPU 数量:", device_count)
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        try:
            # 获取未校正的 ECC 错误（Total, Uncorrected）
            uncorrected = pynvml.nvmlDeviceGetTotalEccErrors(
                handle,
                pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                pynvml.NVML_VOLATILE_ECC
            )
        except pynvml.NVMLError_NotSupported:
            print(f"GPU {i}: 不支持 ECC 统计，可能不是数据中心卡")
            continue
        except pynvml.NVMLError as err:
            print(f"GPU {i}: 无法获取 ECC 计数，错误：{err}")
            continue

        print(f"GPU {i}: 未校正 ECC 错误数 = {uncorrected}")
        if uncorrected > 0:
            bad_gpus.append(i)

    pynvml.nvmlShutdown()
    return bad_gpus

if __name__ == "__main__":
    bad = check_gpu_ecc()
    if bad:
        print("🚨 检测到有问题的 GPU 索引：", bad)
    else:
        print("✅ 所有 GPU 均无未校正 ECC 错误。")
