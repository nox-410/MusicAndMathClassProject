# 音乐与数学 课程项目

## 安装依赖

```shell
apt-get install timidity
pip install PyTuning
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

## 频率生成

* 除12平均律外，必须以原曲的do音为**生律原点**！
    * 换句话说，不存在一个通用的五度相生律或纯律的频率表，必须根据原曲的调号来定
    * 由于MIDI是为12平均律设计的，转调不影响相对音高，所以不记录调号，需要我们自己判断
