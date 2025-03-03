---
license: other
tags:
- CWS
text:
  token-classification:
    type:
      - cws
    language:
      - zh
---

# msra_cws 中文分词数据集

## 数据集概述
msra_cws MSRA数据集是面向新闻领域的中文分词数据集。

### 数据集简介
本数据集包括训练集（14041）、验证集（3250）、测试集（3453），实体类型包括地点(LOC)、混合(MISC)、组织(ORG)、人名(PER)。

### 数据集的格式和结构
数据格式采用conll标准，数据包括两列，第一列输入句中的词划分以及最后一列中每个词对应的分词标签。一个具体case的例子如下：

```
“	S-CWS
种	B-CWS
菜	E-CWS
，	S-CWS
也	S-CWS
有	S-CWS
烦	B-CWS
恼	E-CWS
，	S-CWS
那	S-CWS
是	S-CWS
累	S-CWS
的	S-CWS
时	B-CWS
候	E-CWS
；	S-CWS
```

## 数据集版权信息

Creative Commons Attribution 4.0 International


## 引用方式
```bib
@inproceedings{levow-2006-third,
    title = "The Third International {C}hinese Language Processing Bakeoff: Word Segmentation and Named Entity Recognition",
    author = "Levow, Gina-Anne",
    booktitle = "Proceedings of the Fifth {SIGHAN} Workshop on {C}hinese Language Processing",
    month = jul,
    year = "2006",
    address = "Sydney, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W06-0115",
    pages = "108--117",
}
```
