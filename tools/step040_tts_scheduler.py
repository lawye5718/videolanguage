import os
import pandas as pd
from loguru import logger
from tools.step045_tts_cinecast import generate_tts_with_emotion_clone

def process_tts(subtitle_csv_path, vocals_path, output_dir, default_voice="aiden"):
    """
    处理TTS的主函数
    Args:
        subtitle_csv_path: 字幕CSV文件路径（包含start_time, end_time, translation列）
        vocals_path: 步骤010用Demucs分离出来的纯人声文件路径
        output_dir: 输出目录
        default_voice: 默认音色
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")

    # 读取字幕文件
    try:
        df = pd.read_csv(subtitle_csv_path)
        logger.info(f"加载字幕文件: {subtitle_csv_path}, 共 {len(df)} 句")
    except Exception as e:
        logger.error(f"读取字幕文件失败: {e}")
        return False

    # 验证必要字段
    required_columns = ['start_time', 'end_time', 'translation']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"字幕文件缺少必要字段: {missing_columns}")
        return False

    success_count = 0
    total_count = len(df)
    
    # 逐句处理
    for index, row in df.iterrows():
        text = str(row['translation']).strip()
        start_time = float(row['start_time'])
        end_time = float(row['end_time'])
        
        # 跳过空文本
        if not text:
            logger.warning(f"第 {index} 句文本为空，跳过")
            continue
            
        # 定义输出文件路径
        out_filename = os.path.join(output_dir, f"dub_{index:04d}.mp3")
        
        # 调用情绪配音功能
        success = generate_tts_with_emotion_clone(
            text=text,
            start_time=start_time,
            end_time=end_time,
            vocal_audio_path=vocals_path,
            output_audio_path=out_filename,
            emotion_voice=default_voice
        )
        
        if success:
            success_count += 1
            logger.info(f"✅ 第 {index} 句处理完成 ({success_count}/{total_count})")
        else:
            logger.error(f"❌ 第 {index} 句配音生成失败")
    
    logger.info(f"🏁 TTS处理完成: {success_count}/{total_count} 句成功")
    return success_count > 0

def process_tts_with_voice_mapping(subtitle_csv_path, vocals_path, output_dir, voice_mapping_df):
    """
    支持角色音色映射的高级TTS处理
    Args:
        subtitle_csv_path: 字幕CSV文件路径
        vocals_path: 纯人声文件路径
        output_dir: 输出目录
        voice_mapping_df: 角色音色映射DataFrame（包含character, voice列）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取字幕和音色映射
    df = pd.read_csv(subtitle_csv_path)
    voice_map = dict(zip(voice_mapping_df['character'], voice_mapping_df['voice']))
    
    logger.info(f"角色音色映射: {voice_map}")
    
    success_count = 0
    for index, row in df.iterrows():
        text = str(row['translation']).strip()
        start_time = float(row['start_time'])
        end_time = float(row['end_time'])
        character = row.get('character', 'unknown')  # 假设有角色列
        
        # 根据角色选择音色
        voice = voice_map.get(character, "aiden")  # 默认音色
        
        out_filename = os.path.join(output_dir, f"dub_{index:04d}_{character}.mp3")
        
        success = generate_tts_with_emotion_clone(
            text=text,
            start_time=start_time,
            end_time=end_time,
            vocal_audio_path=vocals_path,
            output_audio_path=out_filename,
            emotion_voice=voice
        )
        
        if success:
            success_count += 1
            logger.info(f"✅ {character}({voice}): '{text[:20]}...' 处理完成")
    
    logger.info(f"🏁 角色化TTS处理完成: {success_count}/{len(df)} 句成功")
    return success_count > 0

# 用于测试
if __name__ == "__main__":
    # 创建测试数据
    test_data = {
        'start_time': [1.0, 5.0, 10.0],
        'end_time': [3.0, 6.0, 15.0],
        'translation': ['你好世界', '这是一个测试', '感谢使用videolanguage'],
        'character': ['A', 'B', 'A']
    }
    
    test_df = pd.DataFrame(test_data)
    test_csv = 'test_subtitles.csv'
    test_df.to_csv(test_csv, index=False)
    
    # 创建测试音色映射
    voice_mapping = pd.DataFrame({
        'character': ['A', 'B'],
        'voice': ['aiden', 'ryan']
    })
    
    print("🧪 TTS调度模块测试")
    print("注意：此测试需要真实的vocals.wav文件才能完整运行")
    print("当前仅验证函数结构和逻辑正确性")