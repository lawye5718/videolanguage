import os
import subprocess
import shutil
import torch

# =================================================================
# 1. 兼容性空壳函数 (接口对齐层，骗过 do_everything.py 防止报错)
# =================================================================

def init_demucs(*args, **kwargs):
    print("💡 [Demucs 优化版] 采用 CLI 模式，跳过预初始化...")
    return True

def load_model(*args, **kwargs):
    print("💡 [Demucs 优化版] 采用子进程按需加载，跳过预占用显存...")
    return True

def release_model(*args, **kwargs):
    print("💡 [Demucs 优化版] 子进程结束 Mac 显存已自动安全释放...")
    return True

# =================================================================
# 2. 核心业务处理层 (基于 subprocess)
# =================================================================

def separate_audio(folder, model_name="htdemucs_ft", device="auto", progress=None, shifts=5):
    """
    单文件分离逻辑。使用系统子进程直接调用 demucs 命令行。
    """
    print(f"▶️ 准备分离音频: 文件夹={folder}")
    
    # 动态寻找需要分离的文件
    audio_path = os.path.join(folder, "download.mp4")
    if not os.path.exists(audio_path):
        audio_path = os.path.join(folder, "download.wav")
        
    if not os.path.exists(audio_path):
        print(f"⚠️ 在 {folder} 找不到 download.mp4 或 download.wav")
        return None, None

    # Mac MPS 加速
    device_cmd = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"▶️ 调用 Demucs 命令行 (设备: {device_cmd})...")
    cmd = [
        "demucs",
        "-n", model_name,
        "--shifts", str(shifts),
        "--two-stems", "vocals", 
        "-d", device_cmd,
        "-o", folder,
        audio_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Demucs 执行失败: {e}")
        return None, None

    # 找到 Demucs 默认的输出位置 (不要去移动它们！)
    track_name = os.path.splitext(os.path.basename(audio_path))[0]
    demucs_out_dir = os.path.join(folder, model_name, track_name)
    
    gen_vocals = os.path.join(demucs_out_dir, "vocals.wav")
    gen_no_vocals = os.path.join(demucs_out_dir, "no_vocals.wav")
    
    if os.path.exists(gen_vocals) and os.path.exists(gen_no_vocals):
        print(f"✅ 人声分离成功！文件保留在原位: {demucs_out_dir}")
        # 核心修复：直接返回它们在 htdemucs_ft 里的原始绝对路径！
        return gen_vocals, gen_no_vocals
    else:
        print(f"❌ 找不到 Demucs 生成的文件于 {demucs_out_dir}")
        return None, None

def separate_all_audio_under_folder(folder, model_name="htdemucs_ft", device="auto", progress=None, shifts=5):
    """
    兼容主程序的批量处理接口。
    这里的关键是：必须返回 3 个值 (状态码, 人声路径, 背景音路径)，以满足 do_everything 的解包要求。
    """
    vocal_path, instr_path = separate_audio(folder, model_name, device, progress, shifts)
    
    if vocal_path and instr_path:
        # 返回 True 和两个路径，完美对接主程序的 status, vocals_path, _ = ...
        return True, vocal_path, instr_path
    else:
        # 返回 False 和 None，防止抛出 unpack error
        return False, None, None