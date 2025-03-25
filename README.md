# NLP_TEXT_CLASSIFICATION
📌 Project: Text Classification Algorithm This project aims to develop and optimize algorithms for text classification.

选择语言 / Choose Language:
- [中文 README](./Readme/README_zh.md)
- [English README](./Readme/README_en.md)
  
目前的进度：  

NER(Named Entity Recognition)  
对o3-mini 4o进行few-shot
在hpc集群对ds蒸馏版进行lora微调和全量微调实验

github上的日志会比实际进度慢，实验具体数据比较杂乱，未整理后暂时不上传  


需要注意的改动
>精确度测算代码已经进行重写，需要对位置进行测算，原测算代码更改为eval函数，并进行了封装。作用可以打开.py文件直接查看。
>具体改动后的调用请参考sm0.py，这个是spacy函数的sm预训练管道在dataset0上的调用函数。可以直接复制粘贴到你们的代码再根据模型不同进行修改。
