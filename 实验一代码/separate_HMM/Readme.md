

### 文件结构(3.3日终版)
```
hmm_cws/
│
├── data_utils.py   # 数据加载和预处理
├── model.py        # 模型定义和参数处理
├── decode.py       # Viterbi解码和分词
├── evaluate.py     # 模型评估
└── main.py         # 主程序入口
```

---


### 使用方式

1. **训练模型**：
```bash
python main.py --train path/to/train.txt --model my_model.pkl
```

2. **测试模型**：
```bash
python main.py --test path/to/test.txt --model my_model.pkl
```

3. **交互式分词**：
```bash
python main.py --model my_model.pkl
```

### 代码特点

1. **模块化设计**：每个文件负责单一职责
   - `data_utils.py`: 数据处理
   - `model.py`: 模型参数管理
   - `decode.py`: 解码算法
   - `evaluate.py`: 评估指标
   - `main.py`: 主程序逻辑

2. **支持两种运行模式**：
   - 命令行训练/测试
   - 交互式分词演示

3. **完善的错误处理**：
   - 自动处理未登录词（UNK）
   - 概率计算使用对数空间防止数值下溢

4. **模型持久化**：
   - 训练后自动保存模型参数
   - 测试/预测时直接加载已有模型