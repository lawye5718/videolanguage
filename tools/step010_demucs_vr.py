import os
import subprocess
import shutil
import torch
from loguru import logger


def separate_audio(folder, model_name="htdemucs_ft", device="auto", progress=None, shifts=5):
    """
    使用系统子进程直接调用 demucs 命令行，彻底绕过 API 兼容性问题。
    使用 --two-stems vocals 直接输出 人声 和 非人声。
    """
    print(f"▶️ 准备分离音频: 文件夹={folder}")
    
    # 1. 查找目标视频或音频
    audio_path = os.path.join(folder, "download.mp4")
    if not os.path.exists(audio_path):
        audio_path = os.path.join(folder, "download.wav")
        
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"找不到需要分离的音频文件: {audio_path}")

    # 2. 决定计算设备 (Mac MPS)
    device_cmd = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 3. 构建命令行调用 (加入 --two-stems vocals 直接产出两轨)
    print(f"▶️ 正在调用 Demucs 命令行引擎 (设备: {device_cmd})...")
    cmd = [
        "demucs",
        "-n", model_name,
        "--shifts", str(shifts),
        "--two-stems", "vocals", # 关键：只分离出 vocals 和 no_vocals
        "-d", device_cmd,
        "-o", folder,
        audio_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"❌ Demucs 命令行执行失败，请检查环境: {e}")

    # 4. 移动和重命名产出文件，适配下游流程
    # Demucs 默认将生成物放在: folder/htdemucs_ft/download/vocals.wav
    track_name = os.path.splitext(os.path.basename(audio_path))[0] # "download"
    demucs_out_dir = os.path.join(folder, model_name, track_name)
    
    gen_vocals = os.path.join(demucs_out_dir, "vocals.wav")
    gen_no_vocals = os.path.join(demucs_out_dir, "no_vocals.wav")
    
    final_vocals = os.path.join(folder, "vocals.wav")
    final_no_vocals = os.path.join(folder, "no_vocals.wav")
    
    if os.path.exists(gen_vocals) and os.path.exists(gen_no_vocals):
        shutil.move(gen_vocals, final_vocals)
        shutil.move(gen_no_vocals, final_no_vocals)
        print(f"✅ 人声分离完美成功！文件保存在: {folder}")
    else:
        raise FileNotFoundError(f"❌ 找不到 Demucs 生成的文件于 {demucs_out_dir}")

    return final_vocals, final_no_vocals


def separate_all_audio_under_folder(root_folder, model_name="htdemucs_ft", device="auto", progress=None, shifts=5):
    """
    对文件夹内所有视频进行音频分离
    """
    logger.info(f'开始处理文件夹: {root_folder}')
    
    # 遍历所有子文件夹
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            try:
                logger.info(f'处理文件夹: {folder_name}')
                separate_audio(folder_path, model_name, device, progress, shifts)
            except Exception as e:
                logger.error(f'处理文件夹 {folder_name} 失败: {e}')
                continue
    
    logger.info('所有音频分离完成')