NLP_TEXT_CLASSIFICATION
选择语言 / Choose Language:
- [中文 README](README_zh.md)
- [English README](README_en.md)
文件说明:
sm.py为使用spacy的en_core_web_sm管道进行训练微调并验证的python代码文件
dataset文件夹内的文件为处理后的数据集，格式为n行两列，"sentence"列中为String类型的句子，"entities"列中为一个嵌套数组，格式为[["实体一",(起始位置，终止位置),"实体一标签"],["实体二",(起始位置，终止位置),"实体二标签"],...]
dataset/deepseek-r1内为调用deepseek-r1的文件，出于隐私考虑,api.py被忽略跟踪没有上传。
trf.py为使用spacy的en_core_web_trf管道进行训练微调并验证的python代码文件，继承了sm.py的格式
trf1.py为使用spacy的en_core_web_trf管道进行训练微调并验证的python代码文件，重构了格式（由于使用trf.py训练精度过低，所以代码进行了重写）
error_result为临时文件，可能是trf.py的预测错误内容
other_model内为NuNER模型
