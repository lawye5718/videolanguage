#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoLanguage 依赖验证测试脚本
在新的 conda 环境中验证所有核心依赖
"""

import os
import sys
import json
from pathlib import Path
from loguru import logger

# 设置日志
logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
logger.add("dependency_test.log", rotation="100 MB", level="DEBUG")

def test_dependencies():
    """测试所有核心依赖"""
    logger.info("🧪 开始依赖验证测试...")
    
    results = {}
    
    # 1. 测试 demucs
    try:
        import demucs
        logger.info("✅ demucs 导入成功")
        logger.info(f"   版本: {getattr(demucs, '__version__', '未知')}")
        results['demucs'] = True
    except Exception as e:
        logger.error(f"❌ demucs 导入失败: {e}")
        results['demucs'] = False
    
    # 2. 测试 whisperx
    try:
        import whisperx
        logger.info("✅ whisperx 导入成功")
        logger.info(f"   版本: {getattr(whisperx, '__version__', '未知')}")
        results['whisperx'] = True
    except Exception as e:
        logger.error(f"❌ whisperx 导入失败: {e}")
        results['whisperx'] = False
    
    # 3. 测试 pyannote.audio
    try:
        import pyannote.audio
        logger.info("✅ pyannote.audio 导入成功")
        logger.info(f"   版本: {getattr(pyannote.audio, '__version__', '未知')}")
        results['pyannote'] = True
    except Exception as e:
        logger.error(f"❌ pyannote.audio 导入失败: {e}")
        results['pyannote'] = False
    
    # 4. 测试 torch 和 MPS 支持
    try:
        import torch
        logger.info("✅ torch 导入成功")
        logger.info(f"   版本: {torch.__version__}")
        logger.info(f"   CUDA 可用: {torch.cuda.is_available()}")
        logger.info(f"   MPS 可用: {torch.backends.mps.is_available()}")
        results['torch'] = True
    except Exception as e:
        logger.error(f"❌ torch 导入失败: {e}")
        results['torch'] = False
    
    # 5. 测试 torchaudio
    try:
        import torchaudio
        logger.info("✅ torchaudio 导入成功")
        logger.info(f"   版本: {torchaudio.__version__}")
        results['torchaudio'] = True
    except Exception as e:
        logger.error(f"❌ torchaudio 导入失败: {e}")
        results['torchaudio'] = False
    
    # 6. 测试 librosa
    try:
        import librosa
        logger.info("✅ librosa 导入成功")
        logger.info(f"   版本: {librosa.__version__}")
        results['librosa'] = True
    except Exception as e:
        logger.error(f"❌ librosa 导入失败: {e}")
        results['librosa'] = False
    
    # 7. 测试 numpy
    try:
        import numpy
        logger.info("✅ numpy 导入成功")
        logger.info(f"   版本: {numpy.__version__}")
        results['numpy'] = True
    except Exception as e:
        logger.error(f"❌ numpy 导入失败: {e}")
        results['numpy'] = False
    
    # 8. 测试 soundfile
    try:
        import soundfile
        logger.info("✅ soundfile 导入成功")
        logger.info(f"   版本: {getattr(soundfile, '__version__', '未知')}")
        results['soundfile'] = True
    except Exception as e:
        logger.error(f"❌ soundfile 导入失败: {e}")
        results['soundfile'] = False
    
    # 9. 测试 ffmpeg 可用性
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            logger.info("✅ ffmpeg 可用")
            logger.info(f"   版本: {version_line}")
            results['ffmpeg'] = True
        else:
            logger.error("❌ ffmpeg 不可用")
            results['ffmpeg'] = False
    except Exception as e:
        logger.error(f"❌ ffmpeg 检查失败: {e}")
        results['ffmpeg'] = False
    
    # 10. 测试系统信息
    logger.info("💻 系统信息:")
    logger.info(f"   Python: {sys.version}")
    logger.info(f"   平台: {sys.platform}")
    logger.info(f"   架构: {os.uname().machine if hasattr(os, 'uname') else 'Unknown'}")
    
    return results

def generate_report(results):
    """生成测试报告"""
    logger.info("📋 生成依赖测试报告...")
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    report = {
        "timestamp": "2026-02-28 05:30:00",
        "environment": "videolang (conda)",
        "python_version": sys.version,
        "test_results": results,
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": f"{success_rate:.1f}%"
        }
    }
    
    # 保存报告
    report_file = Path("dependency_test_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info("📊 依赖测试结果:")
    for dep, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"   {dep}: {status}")
    
    logger.info(f"📈 总体成功率: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    logger.info(f"📄 详细报告已保存到: {report_file}")
    
    return report

def main():
    """主函数"""
    logger.info("🚀 VideoLanguage 依赖验证测试开始")
    logger.info("环境: videolang (conda-forge)")
    logger.info("Python: 3.10.19 (ARM64)")
    
    # 执行测试
    results = test_dependencies()
    
    # 生成报告
    report = generate_report(results)
    
    # 返回测试结果
    success_rate = float(report['summary']['success_rate'].rstrip('%'))
    if success_rate >= 80:
        logger.success("🎉 依赖验证基本通过！可以开始视频处理测试")
        return True
    else:
        logger.error("❌ 依赖验证失败较多，请检查安装")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)