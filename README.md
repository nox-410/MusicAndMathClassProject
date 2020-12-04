# 音乐与数学 课程项目

## 环境配置

* 由于一些软件和库只受特定平台支持，建议在Ubuntu 18.04或更高版本上运行

* 安装TiMidity
```shell
apt install -y timidity
```

* 确保Python版本不低于3.7，并安装下面的库（PyTorch不一定要安装gpu版本）
```shell
PyTuning
torch==1.7.0
torchaudio==0.7.0
```

## TiMidity使用说明

TiMidity是一个强大的midi转音频工具，这里我们只介绍它的最基本的用法
```shell
timidity -Ow -Z <freqtable> -o <outfile> <infile>
```
* `-Ow`：指定生成的格式为wav
* `-Z <freqtable>`：指定频率表，格式如下
```python
# A freqency table contains 128 integers, in mHz.
# The first notes C-1 and the last notes G9.
# lines 1-67 snipped...
391995
415305
# next is A4
440000  
466164
493883
# lines 73-128 snipped...
```
* `-o <outfile> <infile>`：指定输出、输入文件

## 生成频率表
* TODO

## 生成数据集

### 从MIDI生成WAV
* 在`python/gen_wav.py`中设置
```python
MIDI_PATH = './clean_midi'        # midi数据集目录
SAVE_PATH = './dataset'           # wav保存目录
FREQ_TABLE_PATH = './freq_table'  # 频率表目录，需包含3个频率表
SAMPLES = 100 * 3                 # 每种律制生成100首
PARALLEL = 8                      # 8进程并行
```
* 运行
```shell
python3 python/gen_wav.py
```

### 将WAV转换为频谱图（Spectrogram） 
* 在`python/gen_spec.py`中设置
```python
N_FFT = 4096                 # STFT中的bin数量
SAMPLE_RATE = 8000           # 输出采样率
OVERRIDE_PATH = './dataset'  # 将此目录下的wav替换为npy
```
* 运行
```shell
python3 python/gen_spec.py
```

### 可视化频谱图

```shell
python3 python/show_spec.py dataset/pure/08361.npy output.png
```

## 训练
* 先手动将上述生成的`dataset/`分为`train_set/`和`test_set/`
* 运行
```shell
python3 python/train.py
```
