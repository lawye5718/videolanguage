import json
import os
import time
import traceback
import shutil  # <--- 新增这一行

import torch
from loguru import logger
from .step000_video_downloader import get_info_list_from_url, download_single_video, get_target_folder
from .step010_demucs_vr import separate_all_audio_under_folder, init_demucs, load_model, release_model
from .step020_asr import transcribe_all_audio_under_folder
from .step021_asr_whisperx import init_whisperx, init_diarize
from .step022_asr_funasr import init_funasr
from .step030_translation import translate_all_transcript_under_folder
from .step040_tts import generate_all_wavs_under_folder
# 注释掉不需要的TTS模块导入
# from .step042_tts_xtts import init_TTS
# from .step043_tts_cosyvoice import init_cosyvoice
from .step050_synthesize_video import synthesize_all_video_under_folder
from concurrent.futures import ThreadPoolExecutor, as_completed

# 跟踪模型初始化状态
models_initialized = {
    'demucs': False,
    # 'xtts': False,  # 注释掉XTTS
    # 'cosyvoice': False,  # 注释掉CosyVoice
    'whisperx': False,
    'diarize': False,
    'funasr': False
}


def get_available_gpu_memory():
    """获取当前可用的GPU显存大小（GB）"""
    try:
        if torch.cuda.is_available():
            # 获取当前设备的可用显存
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            return free_memory / (1024 ** 3)  # 转换为GB
        return 0  # 如果没有GPU或CUDA不可用
    except Exception:
        return 0  # 出错时返回0


def initialize_models(tts_method, asr_method, diarization):
    """
    初始化所需的模型。
    只在第一次调用时初始化模型，避免重复加载。
    """
    # 使用全局状态跟踪已初始化的模型
    global models_initialized

    with ThreadPoolExecutor() as executor:
        try:
            # Demucs模型初始化
            if not models_initialized['demucs']:
                executor.submit(init_demucs)
                models_initialized['demucs'] = True
                logger.info("Demucs模型初始化完成")
            else:
                logger.info("Demucs模型已初始化，跳过")

            # TTS模型初始化 - 只保留Cinecast和EdgeTTS
            # if tts_method == 'xtts' and not models_initialized['xtts']:
            #     executor.submit(init_TTS)
            #     models_initialized['xtts'] = True
            #     logger.info("XTTS模型初始化完成")
            # elif tts_method == 'cosyvoice' and not models_initialized['cosyvoice']:
            #     executor.submit(init_cosyvoice)
            #     models_initialized['cosyvoice'] = True
            #     logger.info("CosyVoice模型初始化完成")
            # 注：我们现在使用Cinecast API进行TTS，无需本地模型初始化

            # ASR模型初始化
            if asr_method == 'WhisperX':
                if not models_initialized['whisperx']:
                    executor.submit(init_whisperx)
                    models_initialized['whisperx'] = True
                    logger.info("WhisperX模型初始化完成")
                if diarization and not models_initialized['diarize']:
                    executor.submit(init_diarize)
                    models_initialized['diarize'] = True
                    logger.info("Diarize模型初始化完成")
            elif asr_method == 'FunASR' and not models_initialized['funasr']:
                executor.submit(init_funasr)
                models_initialized['funasr'] = True
                logger.info("FunASR模型初始化完成")

        except Exception as e:
            stack_trace = traceback.format_exc()
            logger.error(f"初始化模型失败: {str(e)}\n{stack_trace}")
            # 出现错误时，重置初始化状态
            models_initialized = {key: False for key in models_initialized}
            release_model()  # 释放已加载的模型
            raise


def process_video(info, root_folder, resolution,
                  demucs_model, device, shifts,
                  asr_method, whisper_model, batch_size, diarization, whisper_min_speakers, whisper_max_speakers,
                  translation_method, translation_target_language,
                  tts_method, tts_target_language, voice,
                  subtitles, speed_up, fps, background_music, bgm_volume, video_volume,
                  target_resolution, max_retries, progress_callback=None):
    """
    处理单个视频的完整流程，增加了进度回调函数

    Args:
        progress_callback: 回调函数，用于报告进度和状态，格式为 progress_callback(progress_percent, status_message)
    """
    local_time = time.localtime()

    # 定义进度阶段和权重
    stages = [
        ("下载视频...", 10),  # 10%
        ("人声分离...", 15),  # 15%
        ("AI智能语音识别...", 20),  # 20%
        ("字幕翻译...", 25),  # 25%
        ("AI语音合成...", 20),  # 20%
        ("视频合成...", 10)  # 10%
    ]

    current_stage = 0
    progress_base = 0

    # 报告初始进度
    if progress_callback:
        progress_callback(0, "准备处理...")

    for retry in range(max_retries):
        try:
            # 报告进入下载阶段
            stage_name, stage_weight = stages[current_stage]
            if progress_callback:
                progress_callback(progress_base, stage_name)

            if isinstance(info, str) and info.endswith('.mp4'):
                folder = os.path.dirname(info)
                # os.rename(info, os.path.join(folder, 'download.mp4'))
            else:
                folder = get_target_folder(info, root_folder)
                if folder is None:
                    error_msg = f'无法获取视频目标文件夹: {info["title"]}'
                    logger.warning(error_msg)
                    return False, None, error_msg

                folder = download_single_video(info, root_folder, resolution)
                if folder is None:
                    error_msg = f'下载视频失败: {info["title"]}'
                    logger.warning(error_msg)
                    return False, None, error_msg

            logger.info(f'处理视频: {folder}')

            # 完成下载阶段，进入人声分离阶段
            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback:
                progress_callback(progress_base, stage_name)

            try:
                status, vocals_path, _ = separate_all_audio_under_folder(
                    folder, model_name=demucs_model, device=device, progress=True, shifts=shifts)
                logger.info(f'人声分离完成: {vocals_path}')
                
                # ==========================================
                # 桥接修复：确保ASR能找到分离出的人声文件
                # ==========================================
                if vocals_path and os.path.exists(vocals_path):
                    # 创建安全的人声文件路径（在根目录下）
                    safe_vocals_path = os.path.join(folder, "audio_vocals.wav")
                    if os.path.abspath(vocals_path) != os.path.abspath(safe_vocals_path):
                        import shutil
                        shutil.copy(vocals_path, safe_vocals_path)
                        print(f"🔗 [桥接修复] 已将纯人声复制到根目录供 ASR 识别: {safe_vocals_path}")
                else:
                    logger.error(f"❌ 找不到分离出的人声，无法进行识别！路径: {vocals_path}")
                    return False, None, "人声分离文件不存在"
            except Exception as e:
                stack_trace = traceback.format_exc()
                error_msg = f'人声分离失败: {str(e)}\n{stack_trace}'
                logger.error(error_msg)
                return False, None, error_msg

            # 完成人声分离阶段，进入语音识别阶段
            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback:
                progress_callback(progress_base, stage_name)

            try:
                status, result_json = transcribe_all_audio_under_folder(
                    folder, asr_method=asr_method, whisper_model_name=whisper_model, device=device,
                    batch_size=batch_size, diarization=diarization,
                    min_speakers=whisper_min_speakers,
                    max_speakers=whisper_max_speakers)
                logger.info(f'语音识别完成: {status}')
            except Exception as e:
                stack_trace = traceback.format_exc()
                error_msg = f'语音识别失败: {str(e)}\n{stack_trace}'
                logger.error(error_msg)
                return False, None, error_msg

            # 完成语音识别阶段，进入翻译阶段
            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback:
                progress_callback(progress_base, stage_name)

            try:
                status, summary, translation = translate_all_transcript_under_folder(
                    folder, method=translation_method, target_language=translation_target_language)
                logger.info(f'翻译完成: {status}')
            except Exception as e:
                stack_trace = traceback.format_exc()
                error_msg = f'翻译失败: {str(e)}\n{stack_trace}'
                logger.error(error_msg)
                return False, None, error_msg

            # 完成翻译阶段，进入语音合成阶段
            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback:
                progress_callback(progress_base, stage_name)

            try:
                status, synth_path, _ = generate_all_wavs_under_folder(
                    folder, method=tts_method, target_language=tts_target_language, voice=voice)
                logger.info(f'语音合成完成: {synth_path}')
            except Exception as e:
                stack_trace = traceback.format_exc()
                error_msg = f'语音合成失败: {str(e)}\n{stack_trace}'
                logger.error(error_msg)
                return False, None, error_msg

            # 完成语音合成阶段，进入视频合成阶段
            current_stage += 1
            progress_base += stage_weight
            stage_name, stage_weight = stages[current_stage]
            if progress_callback:
                progress_callback(progress_base, stage_name)

            try:
                status, output_video = synthesize_all_video_under_folder(
                    folder, subtitles=subtitles, speed_up=speed_up, fps=fps, resolution=target_resolution,
                    background_music=background_music, bgm_volume=bgm_volume, video_volume=video_volume)
                logger.info(f'视频合成完成: {output_video}')
            except Exception as e:
                stack_trace = traceback.format_exc()
                error_msg = f'视频合成失败: {str(e)}\n{stack_trace}'
                logger.error(error_msg)
                return False, None, error_msg

            # 完成所有阶段，报告100%进度
            if progress_callback:
                progress_callback(100, "处理完成!")

            return True, output_video, "处理成功"
        except Exception as e:
            stack_trace = traceback.format_exc()
            error_msg = f'处理视频时发生错误 {info["title"] if isinstance(info, dict) else info}: {str(e)}\n{stack_trace}'
            logger.error(error_msg)
            if retry < max_retries - 1:
                logger.info(f'尝试重试 {retry + 2}/{max_retries}...')
            else:
                return False, None, error_msg

    return False, None, f"达到最大重试次数: {max_retries}"


def do_everything(root_folder, url, num_videos=5, resolution='1080p',
                  demucs_model='htdemucs_ft', device='auto', shifts=5,
                  asr_method='WhisperX', whisper_model='large', batch_size=32, diarization=False,
                  whisper_min_speakers=None, whisper_max_speakers=None,
                  translation_method='LLM', translation_target_language='简体中文',
                  tts_method='xtts', tts_target_language='中文', voice='zh-CN-XiaoxiaoNeural',
                  subtitles=True, speed_up=1.00, fps=30,
                  background_music=None, bgm_volume=0.5, video_volume=1.0, target_resolution='1080p',
                  max_workers=3, max_retries=5, progress_callback=None):
    """
    处理整个视频处理流程，增加了进度回调函数

    Args:
        progress_callback: 回调函数，用于报告进度和状态，格式为 progress_callback(progress_percent, status_message)
    """
    try:
        success_list = []
        fail_list = []
        error_details = []
        
        # ==========================================
        # 强制安全锁：无视界面传入的任何旧设置，强制使用 Cinecast
        tts_method = "Cinecast"  # <--- 新增这行代码！
        # ==========================================

        # 记录处理开始信息和所有参数
        logger.info("-" * 50)
        logger.info(f"开始处理任务: {url}")
        logger.info(f"参数: 输出文件夹={root_folder}, 视频数量={num_videos}, 分辨率={resolution}")
        logger.info(f"人声分离: 模型={demucs_model}, 设备={device}, 移位次数={shifts}")
        logger.info(f"语音识别: 方法={asr_method}, 模型={whisper_model}, 批大小={batch_size}")
        logger.info(f"翻译: 方法={translation_method}, 目标语言={translation_target_language}")
        logger.info(f"语音合成: 方法={tts_method}, 目标语言={tts_target_language}, 声音={voice}")
        logger.info(f"视频合成: 字幕={subtitles}, 速度={speed_up}, FPS={fps}, 分辨率={target_resolution}")
        logger.info("-" * 50)

        url = url.replace(' ', '').replace('，', '\n').replace(',', '\n')
        urls = [_ for _ in url.split('\n') if _]

        # 初始化模型（改用新的初始化函数）
        try:
            if progress_callback:
                progress_callback(5, "初始化模型中...")
            initialize_models(tts_method, asr_method, diarization)
        except Exception as e:
            stack_trace = traceback.format_exc()
            logger.error(f"初始化模型失败: {str(e)}\n{stack_trace}")
            return f"初始化模型失败: {str(e)}", None

        out_video = None
        # 1. 确定视频源是网络还是本地
        is_local_file = os.path.exists(url) and not url.startswith("http")
        
        if is_local_file:
            logger.info(f"✅ 检测到本地视频文件: {url}")
            original_file_path = url  # 绝对路径，直接使用
            
            # 获取原始视频文件名（不带路径）
            original_file_name = os.path.basename(url)
            
            # 去除文件扩展名，生成文件夹名称
            new_folder_name = os.path.splitext(original_file_name)[0]
            
            # 构建新文件夹的完整路径
            new_folder_path = os.path.join(root_folder, new_folder_name)
            
            # 在 root_folder 下创建该文件夹
            os.makedirs(new_folder_path, exist_ok=True)
            
            # 构建新位置的完整路径
            new_file_path = os.path.join(new_folder_path, "download.mp4")
            
            # 将视频文件拷贝到新创建的文件夹中并重命名
            try:
                # 如果原始文件就是目标文件，跳过 copy
                if os.path.abspath(original_file_path) != os.path.abspath(new_file_path):
                    # 新增：确保目标文件夹一定存在，如果不存在就自动创建
                    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
                    
                    # 执行拷贝
                    shutil.copy(original_file_path, new_file_path)
                    logger.info(f"✅ 成功将视频复制到工作区: {new_file_path}")
                else:
                    logger.info(f"ℹ️  视频已在正确位置，无需复制")
            except Exception as e:
                logger.error(f"❌ 视频拷贝失败，检查路径。源: {original_file_path}, 目标: {new_file_path}")
                raise e
                
            success, output_video, error_msg = process_video(
                new_file_path, root_folder, resolution,
                demucs_model, device, shifts,
                asr_method, whisper_model, batch_size, diarization, whisper_min_speakers, whisper_max_speakers,
                translation_method, translation_target_language,
                tts_method, tts_target_language, voice,
                subtitles, speed_up, fps, background_music, bgm_volume, video_volume,
                target_resolution, max_retries, progress_callback
            )
            
            if success:
                logger.info(f"✅ 视频处理成功: {new_file_path}")
                return '处理成功', output_video
            else:
                logger.error(f"❌ 视频处理失败: {new_file_path}, 错误: {error_msg}")
                return f'处理失败: {error_msg}', None
        else:
            try:
                videos_info = []
                if progress_callback:
                    progress_callback(10, "获取视频信息中...")

                for video_info in get_info_list_from_url(urls, num_videos):
                    videos_info.append(video_info)

                if not videos_info:
                    return "获取视频信息失败，请检查URL是否正确", None

                for info in videos_info:
                    try:
                        success, output_video, error_msg = process_video(
                            info, root_folder, resolution,
                            demucs_model, device, shifts,
                            asr_method, whisper_model, batch_size, diarization, whisper_min_speakers,
                            whisper_max_speakers,
                            translation_method, translation_target_language,
                            tts_method, tts_target_language, voice,
                            subtitles, speed_up, fps, background_music, bgm_volume, video_volume,
                            target_resolution, max_retries, progress_callback
                        )

                        if success:
                            success_list.append(info)
                            out_video = output_video
                            logger.info(f"成功处理视频: {info['title'] if isinstance(info, dict) else info}")
                        else:
                            fail_list.append(info)
                            error_details.append(f"{info['title'] if isinstance(info, dict) else info}: {error_msg}")
                            logger.error(
                                f"处理视频失败: {info['title'] if isinstance(info, dict) else info}, 错误: {error_msg}")
                    except Exception as e:
                        stack_trace = traceback.format_exc()
                        fail_list.append(info)
                        error_details.append(f"{info['title'] if isinstance(info, dict) else info}: {str(e)}")
                        logger.error(
                            f"处理视频出错: {info['title'] if isinstance(info, dict) else info}, 错误: {str(e)}\n{stack_trace}")
            except Exception as e:
                stack_trace = traceback.format_exc()
                logger.error(f"获取视频列表失败: {str(e)}\n{stack_trace}")
                return f"获取视频列表失败: {str(e)}", None

        # 记录处理结果汇总
        logger.info("-" * 50)
        logger.info(f"处理完成: 成功={len(success_list)}, 失败={len(fail_list)}")
        if error_details:
            logger.info("失败详情:")
            for detail in error_details:
                logger.info(f"  - {detail}")

        return f'成功: {len(success_list)}\n失败: {len(fail_list)}', out_video

    except Exception as e:
        # 捕获整体处理过程中的任何错误
        stack_trace = traceback.format_exc()
        error_msg = f"处理过程中发生错误: {str(e)}\n{stack_trace}"
        logger.error(error_msg)
        return error_msg, None


if __name__ == '__main__':
    do_everything(
        root_folder='videos',
        url='https://www.bilibili.com/video/BV1kr421M7vz/',
        translation_method='LLM',
        # translation_method = 'Google Translate', translation_target_language = '简体中文',
    )