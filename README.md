## 基于高斯过程回归的股票价格预测
### 文件组织说明：
- `configs/`为项目配置文件，其中可以修改的部分为：
  - `Config`字段下的`company`值，目前支持的公司代号为A,AA,ABC,ABCB,ACLS,ACNB,ADBE,ADP,AEG,AIR，项目展示使用的是公司ADBE
  - `GPR/kernel`字段下的核类型，支持的核类型为`RBF`和`DotProduct`
- `data/`文件夹下储存项目数据文件，分为训练集和测试集，在训练模型时使用四年的数据，配置文件中的起始和终止时间也可以修改
- `models/`文件夹下存储高斯过程回归模型和核贝叶斯线性回归（作为基线模型）
- `utils/`文件夹下存储画图和数据预处理相关文件

### 项目运行
```bash
pip install -e .
```
安装相关依赖，之后运行
```bash
python train.py
```
获取模型运行结果。程序运行会创建`images/`文件夹，储存GPR模型和贝叶斯模型在测试集和训练集上的结果，同时控制台展示两种模型对应的均方误差数据。