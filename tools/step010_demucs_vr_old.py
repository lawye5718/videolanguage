import os
import subprocess
import shutil
import torch
from demucs.separate import main as demucs_main, get_parser
import time
from loguru import logger
from .utils import save_wav, normalize_wav


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

# 全局变量
auto_device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
separator = None
model_loaded = False  # 新增标志，跟踪模型是否已加载
current_model_config = {}  # 新增变量，存储当前加载模型的配置


def init_demucs():
    """
    初始化Demucs。
    对于命令行版本，这里只是验证环境。
    """
    global model_loaded
    if not model_loaded:
        logger.info("验证Demucs环境...")
        try:
            result = subprocess.run(['demucs', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("✅ Demucs命令行工具可用")
                model_loaded = True
            else:
                logger.error("❌ Demucs命令行工具不可用")
                raise RuntimeError("Demucs命令行工具不可用")
        except Exception as e:
            logger.error(f"❌ Demucs环境检查失败: {e}")
            raise
    else:
        logger.info("Demucs环境已经验证，跳过初始化")


def load_model(model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True,
               shifts: int = 5):
    """
    返回模型配置信息。
    对于命令行版本，不需要实际加载模型。
    """
    global model_loaded, current_model_config
    
    # 验证环境
    init_demucs()
    
    # 存储当前模型配置
    current_model_config = {
        'model_name': model_name,
        'device': 'auto' if device == 'auto' else device,
        'shifts': shifts
    }
    
    logger.info(f'Demucs配置已设置: {current_model_config}')
    return current_model_config


def release_model():
    """
    释放模型资源，避免内存泄漏
    """
    global separator, model_loaded, current_model_config

    if separator is not None:
        logger.info('正在释放Demucs模型资源...')
        # 删除引用
        separator = None
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_loaded = False
        current_model_config = {}
        logger.info('Demucs模型资源已释放')


def separate_audio(folder: str, model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True,
                   shifts: int = 5) -> None:
    """
    分离音频文件
    """
    global separator
    audio_path = os.path.join(folder, 'audio.wav')
    if not os.path.exists(audio_path):
        return None, None
    vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
    instruments_output_path = os.path.join(folder, 'audio_instruments.wav')

    if os.path.exists(vocal_output_path) and os.path.exists(instruments_output_path):
        logger.info(f'音频已分离: {folder}')
        return vocal_output_path, instruments_output_path

    logger.info(f'正在分离音频: {folder}')

    try:
        # 确保模型已加载并且配置正确
        if not model_loaded or current_model_config.get('model_name') != model_name or \
                (current_model_config.get('device') == 'auto') != (device == 'auto') or \
                current_model_config.get('shifts') != shifts:
            load_model(model_name, device, progress, shifts)

        t_start = time.time()

        try:
            origin, separated = separator.separate_audio_file(audio_path)
        except Exception as e:
            logger.error(f'音频分离出错: {e}')
            # 在发生错误时尝试重新加载模型一次
            release_model()
            load_model(model_name, device, progress, shifts)
            logger.info(f'已重新加载模型，重试分离...')
            origin, separated = separator.separate_audio_file(audio_path)

        t_end = time.time()
        logger.info(f'音频分离完成，用时 {t_end - t_start:.2f} 秒')

        vocals = separated['vocals'].numpy().T
        instruments = None
        for k, v in separated.items():
            if k == 'vocals':
                continue
            if instruments is None:
                instruments = v
            else:
                instruments += v
        instruments = instruments.numpy().T

        save_wav(vocals, vocal_output_path, sample_rate=44100)
        logger.info(f'已保存人声: {vocal_output_path}')

        save_wav(instruments, instruments_output_path, sample_rate=44100)
        logger.info(f'已保存伴奏: {instruments_output_path}')

        return vocal_output_path, instruments_output_path

    except Exception as e:
        logger.error(f'分离音频失败: {str(e)}')
        # 出现错误，释放模型资源并重新抛出异常
        release_model()
        raise


def extract_audio_from_video(folder: str) -> bool:
    """
    从视频中提取音频
    """
    video_path = os.path.join(folder, 'download.mp4')
    if not os.path.exists(video_path):
        return False
    audio_path = os.path.join(folder, 'audio.wav')
    if os.path.exists(audio_path):
        logger.info(f'音频已提取: {folder}')
        return True
    logger.info(f'正在从视频提取音频: {folder}')

    os.system(
        f'ffmpeg -loglevel error -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{audio_path}"')

    time.sleep(1)
    logger.info(f'音频提取完成: {folder}')
    return True


def separate_all_audio_under_folder(root_folder: str, model_name: str = "htdemucs_ft", device: str = 'auto',
                                    progress: bool = True, shifts: int = 5) -> None:
    """
    分离文件夹下所有音频
    """
    global separator
    vocal_output_path, instruments_output_path = None, None

    try:
        for subdir, dirs, files in os.walk(root_folder):
            if 'download.mp4' not in files:
                continue
            if 'audio.wav' not in files:
                extract_audio_from_video(subdir)
            if 'audio_vocals.wav' not in files:
                vocal_output_path, instruments_output_path = separate_audio(subdir, model_name, device, progress,
                                                                            shifts)
            elif 'audio_vocals.wav' in files and 'audio_instruments.wav' in files:
                vocal_output_path = os.path.join(subdir, 'audio_vocals.wav')
                instruments_output_path = os.path.join(subdir, 'audio_instruments.wav')
                logger.info(f'音频已分离: {subdir}')

        logger.info(f'已完成所有音频分离: {root_folder}')
        return f'所有音频分离完成: {root_folder}', vocal_output_path, instruments_output_path

    except Exception as e:
        logger.error(f'分离音频过程中出错: {str(e)}')
        # 出现任何错误，释放模型资源
        release_model()
        raise


def clear_memory():
    """
    清理内存和显存（跨平台兼容）
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()  # 强制释放 Mac 统一内存

if __name__ == '__main__':
    folder = r"videos"
    separate_all_audio_under_folder(folder, shifts=0)