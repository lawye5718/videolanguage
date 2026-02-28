#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版 Demucs 人声分离工具
适配 demucs 4.0.1 版本
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from loguru import logger

def separate_audio_simple(input_audio_path, output_dir, model_name="htdemucs_ft"):
    """
    简化版音频分离函数
    使用 demucs 命令行接口
    """
    logger.info(f"🎵 开始人声分离: {input_audio_path}")
    
    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 构建 demucs 命令
    cmd = [
        "demucs",
        "-n", model_name,  # 模型名称
        "--two-stems", "vocals",  # 只分离人声和伴奏
        "-o", str(output_dir),  # 输出目录
        str(input_audio_path)  # 输入音频文件
    ]
    
    logger.info(f"🔄 执行命令: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5分钟超时
        end_time = time.time()
        
        if result.returncode == 0:
            logger.success(f"✅ 人声分离完成! 耗时: {end_time - start_time:.2f}秒")
            
            # 查找输出文件
            model_output_dir = output_dir / model_name
            vocals_file = model_output_dir / "vocals.wav"
            other_file = model_output_dir / "no_vocals.wav"
            
            if vocals_file.exists():
                logger.info(f"   人声轨道: {vocals_file}")
                vocals_size = vocals_file.stat().st_size
                logger.info(f"   文件大小: {vocals_size / (1024*1024):.2f} MB")
                
                # 重命名文件以便后续处理
                final_vocals = output_dir.parent / "audio_vocals.wav"
                final_other = output_dir.parent / "audio_instruments.wav"
                
                vocals_file.rename(final_vocals)
                if other_file.exists():
                    other_file.rename(final_other)
                    logger.info(f"   伴奏轨道: {final_other}")
                
                return str(final_vocals), str(final_other) if other_file.exists() else None
            else:
                logger.error("❌ 未找到输出文件")
                return None, None
        else:
            logger.error(f"❌ 分离失败: {result.stderr}")
            return None, None
            
    except subprocess.TimeoutExpired:
        logger.error("❌ 分离超时 (超过5分钟)")
        return None, None
    except Exception as e:
        logger.error(f"❌ 分离出错: {e}")
        return None, None

def extract_audio_from_video_ffmpeg(video_path, audio_path):
    """使用FFmpeg从视频提取音频"""
    logger.info(f"🔄 从视频提取音频: {video_path}")
    
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vn",  # 不包含视频
        "-acodec", "pcm_s16le",  # PCM 16位
        "-ar", "44100",  # 采样率44.1kHz
        "-ac", "2",  # 立体声
        str(audio_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            logger.success(f"✅ 音频提取完成: {audio_path}")
            return True
        else:
            logger.error(f"❌ 音频提取失败: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ 音频提取出错: {e}")
        return False

def separate_all_audio_under_folder(root_folder, model_name="htdemucs_ft"):
    """
    分离文件夹下所有音频的主要函数
    返回: (是否成功, 人声路径, 伴奏路径)
    """
    logger.info(f"🎬 开始处理文件夹: {root_folder}")
    
    root_path = Path(root_folder)
    
    # 查找视频文件
    video_files = list(root_path.glob("*.mp4")) + list(root_path.glob("*.mov")) + list(root_path.glob("*.avi"))
    
    if not video_files:
        logger.warning("❌ 未找到视频文件")
        return False, None, None
    
    video_path = video_files[0]  # 使用第一个视频文件
    logger.info(f"🎥 使用视频: {video_path.name}")
    
    # 提取音频
    audio_path = root_path / "audio.wav"
    if not audio_path.exists():
        if not extract_audio_from_video_ffmpeg(video_path, audio_path):
            return False, None, None
    else:
        logger.info("✅ 音频已存在，跳过提取")
    
    # 执行人声分离
    temp_output_dir = root_path / "demucs_output"
    vocal_path, instrumental_path = separate_audio_simple(audio_path, temp_output_dir, model_name)
    
    # 清理临时目录
    if temp_output_dir.exists():
        import shutil
        shutil.rmtree(temp_output_dir)
    
    if vocal_path:
        logger.success("🎉 全链路人声分离完成!")
        return True, vocal_path, instrumental_path
    else:
        logger.error("❌ 人声分离失败")
        return False, None, None

if __name__ == "__main__":
    # 测试代码
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        success, vocal_path, instrumental_path = separate_all_audio_under_folder(folder_path)
        print(f"结果: {success}")
        if vocal_path:
            print(f"人声: {vocal_path}")
        if instrumental_path:
            print(f"伴奏: {instrumental_path}")
    else:
        print("用法: python step010_demucs_simple.py <文件夹路径>")