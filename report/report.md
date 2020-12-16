# 《音乐与数学》期末研究题 : 三种律制的实现·机器自动判别

1700012741	梁书豪	信息科学技术学院			1700012728	石屹宁	信息科学技术学院
1700012909	宋苑铭	信息科学技术学院			1700011310	王天冶	物理学院

---

## 一. 背景与介绍

​		律制是规定两个相差八度音程之间的音的精确频率的数学方法。常见律制有十二平均律、纯律和五度相生律等。然而，律制与律制之间每个音的差异往往是比较细微的，往往只有几个音分的差距。在这种情况下，即使是对于受过专业训练的人来说，想要人为地判别所听的音乐属于何种律制是比较困难的。

​		不过，随着近年来机器学习，特征工程的不断发展，这些研究成果在音频领域也有了很多应用。因此，有希望利用这些技术，实现机器自动判别音乐的律制。我们在本实验中自己制作了由三种律制组成的原始数据集，并在其上应用了一些学习算法，达到了超过90%的测试正确率。我们也开源了我们的代码，详情请访问https://github.com/nox-410/MusicAndMathClassProject。

## 二. 三种律制的实现

​		我们借助timidity++来实现三种律制，timidity++是一款能够播放midi文件并将其转为wave格式的软件，在转换的过程中，运行我们使用自己定义的频率表，以此实现不同的律制。

​		我们自己制作了十二平均律(equal)，毕达哥拉斯五度相生律(pythagorean)以及纯律(pure)的频率表。

### 1. 十二平均律

我们以每个八度12个半音为一组，可以得到下面的比例关系式，其中1代表C，而2代表高八度的C。
$$
[1, 2^{1/12}, 2^{1/6}, 2^{1/4}, 2^{1/3}, 2^{5/12}, \sqrt{2}, 2^{7/12}, 2^{2/3}, 2^{3/4}, 2^{5/6}, 2^{11/12}, 2]
$$
然后以国际标准音高A=440Hz即可生成每一个音的频率值。

###　2. 五度相生律

同样的，我们可以用公式计算出五度相生律的频率关系如下：
$$
[1, \frac{2187}{2048}, \frac{9}{8}, \frac{19683}{16384}, \frac{81}{64}, \frac{177147}{131072}, \frac{729}{512}, \frac{3}{2}, \frac{6561}{4096}, \frac{27}{16}, \frac{59049}{32768}, \frac{243}{128}, 2]
$$

### 3. 纯律

我们也可以计算出纯律的比值关系如下：
$$
[1, \frac{16}{15}, \frac98, \frac65, \frac54, \frac43, \frac{45}{32}, \frac32, \frac85, \frac53, \frac95, \frac{15}8, 2]
$$
对于纯律和五度相生律，我们总是使用之前十二平均律中的中央C=261.625565Hz作为生律原点(reference note)。

## 三. 机器自动判别



## 四. 实验细节与结果

### 1. 数据准备

​		我们使用的midi数据集来自于 [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd "The Lakh MIDI Dataset v0.1")，这个数据集中包含了45129个midi文件，这足以支持我们的学习算法。
