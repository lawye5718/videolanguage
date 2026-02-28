# VideoLanguage项目Mac移植说明

## 移植概述
本项目已从Windows/CUDA架构移植到Mac mini (Apple Silicon)，主要改动如下：

## 主要修改内容

### 1. 硬件加速适配
- **WhisperX模块** (`tools/step021_asr_whisperx.py`): 
  - 将`cuda`设备检测改为MPS优先的智能检测
  - 添加`float32`精度支持以确保MPS稳定性
  - 保留CPU回退选项

- **FunASR模块** (`tools/step022_asr_funasr.py`):
  - 修改设备检测逻辑，优先使用MPS
  - 简化模型加载参数

- **Demucs模块** (`tools/step010_demucs_vr.py`):
  - 更新全局设备检测
  - 添加跨平台内存清理函数

### 2. Cinecast TTS集成
- **新增模块** (`tools/step045_tts_cinecast.py`):
  - 实现与本地Cinecast API的对接
  - 支持OpenAI兼容的TTS调用格式
  - 流式音频生成

### 3. 依赖管理优化
- **requirements.txt**:
  - 移除CUDA特定依赖
  - 添加标准PyTorch依赖（自动适配MPS）
  - 注释掉可能不兼容的CTranslate2

### 4. 内存管理增强
- **新增工具** (`tools/memory_utils.py`):
  - 跨平台内存清理函数
  - 内存使用状态监控
  - 统一的垃圾回收机制

## 使用说明

### 环境准备
```bash
# 安装FFmpeg（必需）
brew install ffmpeg

# 安装Python依赖
pip install -r requirements.txt

# 确保Cinecast服务运行
# 访问 http://localhost:8888 验证API可用性
```

### 测试Cinecast集成
```bash
cd tools
python step045_tts_cinecast.py
```

### 内存监控
```bash
cd tools
python memory_utils.py
```

## 注意事项

1. **精度设置**: Mac MPS对float16支持有限，项目默认使用float32确保稳定性
2. **内存管理**: Mac采用统一内存架构，需定期清理避免内存溢出
3. **模型兼容性**: 部分CUDA优化的模型可能需要调整参数
4. **性能预期**: MPS性能通常低于高端CUDA显卡，但优于CPU

## 待优化项

- [ ] WhisperX时间戳对齐在MPS上的稳定性测试
- [ ] FunASR在MPS上的性能基准测试
- [ ] Demucs分离效果验证
- [ ] 整体流程的端到端测试

## 故障排除

如遇MPS相关错误：
1. 检查PyTorch版本是否支持MPS
2. 尝试降低batch_size
3. 使用CPU模式作为备选方案
4. 查看系统内存使用情况