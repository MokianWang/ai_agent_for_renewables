# AI Agent for Renewables - Power System Prediction & Optimization

基于机器学习的电力负荷/光伏/风电预测系统，结合 IEEE 30 节点电力系统最优潮流（OPF）调度优化。

---

## 📋 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [安装配置](#安装配置)
- [使用指南](#使用指南)
  - [1. 模型训练与测试](#1-模型训练与测试)
  - [2. OPF 最优潮流调度](#2-opf-最优潮流调度)
- [系统架构](#系统架构)
  - [机器学习预测模块](#机器学习预测模块)
  - [电力系统 OPF 模块](#电力系统-opf-模块)
- [测试结果](#测试结果)
- [代码重构说明](#代码重构说明)
- [故障排除](#故障排除)
- [技术栈](#技术栈)

---

## 项目概述

本项目包含两个核心模块：

1. **预测模块**：使用 LightGBM 对太阳能、风能和电力负荷进行高精度时间序列预测
2. **优化模块**：基于 pandapower 实现 IEEE 30 节点电力系统的最优潮流（OPF）调度，整合可再生能源

### 应用场景

- 可再生能源发电预测
- 电力负荷预测
- 电力系统经济调度
- 电网优化运行
- 微网能量管理

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练预测模型
python src/train.py

# 3. 测试模型性能
python src/test.py

# 4. 运行 OPF 调度
python src/dispatch.py
```

---

## 功能特性

### 预测模块
- ✅ **多类型预测**：支持太阳能、风能、负荷三种预测类型
- ✅ **多区域支持**：覆盖 4 个区域的 9 个预测任务
- ✅ **LightGBM 模型**：高效梯度提升树，精度高速度快
- ✅ **自动评估**：RMSE 指标、可视化对比图
- ✅ **模块化设计**：消除重复代码，易于维护

### OPF 调度模块
- ✅ **IEEE 30 节点系统**：标准测试系统，41 条输电线路
- ✅ **多区域负荷**：4 个独立负荷区域
- ✅ **可再生能源整合**：5 台可再生发电机（3 太阳能 + 2 风能）
- ✅ **时序优化**：96 个时间步（24小时 × 15分钟）
- ✅ **100% 收敛率**：稳定可靠的 OPF 求解
- ✅ **详细输出**：每个时间步的母线、发电机数据

---

## 项目结构

```
ai_agent_for_renewables/
├── src/
│   ├── data_loader.py      # 数据加载、归一化、特征工程
│   ├── utils.py            # 共享工具函数（评估、绘图、常量）
│   ├── train.py            # 模型训练脚本
│   ├── test.py             # 模型测试脚本
│   └── dispatch.py         # OPF 最优潮流调度
├── data/CAISO/             # CAISO 历史数据
├── predictions/            # 预测结果 CSV（供 OPF 使用）
├── models/                 # 训练的 LightGBM 模型
├── results/                # 预测结果对比
├── figures/                # 可视化图表
├── opf_results/            # OPF 调度结果
│   ├── summary.csv         # 成本汇总
│   ├── t_{i}_bus.csv       # 母线数据（每时间步）
│   └── t_{i}_gen.csv       # 发电机数据（每时间步）
├── config.json             # 配置文件
├── requirements.txt        # Python 依赖
├── README.md               # 本文档
├── CODE_STRUCTURE.md       # 代码结构详细说明
└── REFACTORING_SUMMARY.md  # 重构总结
```

---

## 安装配置

### 环境要求

- Python 3.8+
- 推荐：conda 虚拟环境

### 安装步骤

```bash
# 创建虚拟环境（可选）
conda create -n renewables python=3.10
conda activate renewables

# 安装依赖
pip install -r requirements.txt

# 性能加速（可选，推荐）
pip install numba
```

### 主要依赖

```
lightgbm>=3.3.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
pandapower>=2.13.0
joblib>=1.2.0
```

---

## 使用指南

### 1. 模型训练与测试

#### 训练所有模型

```bash
cd src
python train.py
```

将训练 9 个模型：
- **光伏发电** (solar_power): Zone 1, 2, 3
- **风力发电** (wind_power): Zone 3, 4
- **电力负荷** (load_power): Zone 1, 2, 3, 4

#### 测试模型性能

```bash
python test.py
```

输出：
- RMSE（均方根误差）汇总表
- 预测结果 CSV (`results/`)
- 实际值 vs 预测值对比图 (`figures/`)

#### 自定义配置

编辑 `config.json` 调整参数：

```json
{
  "data_path": "data/CAISO",
  "timestep": 15,
  "resample": true,
  "history_hour": 6,
  "test_size": 0.2,
  "random_state": 42
}
```

---

### 2. OPF 最优潮流调度

#### 基本运行

```bash
cd src
python dispatch.py
```

#### 使用自定义预测数据

将预测 CSV 文件放入 `predictions/` 目录：

**必需文件**：
- `load_power_1.csv` ~ `load_power_4.csv`
- `solar_power_1.csv` ~ `solar_power_3.csv`
- `wind_power_3.csv`, `wind_power_4.csv`

**文件格式**：
```csv
Predicted
0.85
0.72
0.91
...
```

#### 输出结果

所有结果保存在 `opf_results/`：

| 文件 | 说明 |
|------|------|
| `summary.csv` | 时间序列成本和收敛状态 |
| `t_{i}_bus.csv` | 母线电压、相角、功率注入（i=1~96） |
| `t_{i}_gen.csv` | 发电机有功、无功、电压（i=1~96） |

---

## 系统架构

### 机器学习预测模块

```
┌─────────────┐
│ 历史数据     │
│ (CAISO CSV) │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ data_loader │ ← 数据加载、归一化、特征工程
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  LightGBM   │ ← 模型训练 (train.py)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  预测结果    │ → predictions/*.csv
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  模型评估    │ ← 测试 (test.py)
│  (utils.py) │ → RMSE, 图表
└─────────────┘
```

### 电力系统 OPF 模块

```
┌──────────────────┐
│ 预测数据输入      │
│ predictions/*.csv│
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ IEEE 30 节点系统  │
│ - 30 buses       │
│ - 41 lines       │
│ - 6 conventional │
│ - 5 renewable    │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ 时序 OPF 优化     │
│ (dispatch.py)    │
│ - 96 time steps  │
│ - pandapower     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ 调度结果          │
│ opf_results/     │
└──────────────────┘
```

---

## 测试结果

### 预测模块性能

| 类型 | Zone | RMSE |
|------|------|------|
| wind_power | 3 | 0.0015 |
| wind_power | 4 | 0.0023 |
| load_power | 3 | 0.0026 |
| load_power | 1 | 0.0033 |
| load_power | 2 | 0.0055 |
| load_power | 4 | 0.0057 |
| solar_power | 2 | 0.0075 |
| solar_power | 3 | 0.0085 |
| solar_power | 1 | 0.0093 |

**平均 RMSE: 0.0051** ⭐

### OPF 调度性能

| 指标 | 数值 |
|------|------|
| 总时间步数 | 96 |
| 收敛率 | **100%** |
| 平均成本 | $740.71 / 时间步 |
| 成本范围 | $241 - $1373 |
| 发电机数量 | 11 (6 常规 + 5 可再生) |
| 执行时间 | ~2-5 分钟 |

---

## 代码重构说明

### 重构目标

消除 `train.py` 和 `test.py` 之间的重复代码，提高可维护性。

### 重构成果

| 文件 | 重构前 | 重构后 | 减少 |
|------|--------|--------|------|
| train.py | 150行 | 110行 | **-27%** |
| test.py | 102行 | 60行 | **-41%** |
| **消除重复** | ~90行 | 0行 | **100%** |

### 新增模块

**`src/utils.py`** - 共享工具函数：
- `evaluate_model()` - 统一模型评估
- `plot_predictions()` - 可复用绘图
- `print_summary_table()` - 格式化表格
- `TASKS` - 集中化任务定义

### 优势

✅ **DRY 原则**：无重复代码  
✅ **单一事实来源**：评估逻辑只在一处  
✅ **易于维护**：bug 修复只需修改一次  
✅ **一致性保证**：训练和测试使用完全相同的评估代码  
✅ **可扩展性**：通过修改 utils.py 即可添加新功能  

详见 [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) 和 [CODE_STRUCTURE.md](CODE_STRUCTURE.md)。

---

## 故障排除

### 问题 1: OPF 不收敛

**症状**：`Convergence rate < 100%`

**解决方案**：
1. 检查预测数据是否正确缩放（0-1 范围）
2. 验证发电机限制不过于严格
3. 确保线路容量充足
4. 尝试调整 `dispatch.py` 中的电压限制

### 问题 2: 执行速度慢

**解决方案**：
```bash
# 安装 numba 加速（5-10倍提升）
pip install numba
```

### 问题 3: 导入错误

**解决方案**：
```bash
# 升级关键依赖
pip install --upgrade pandapower pandas numpy lightgbm
```

### 问题 4: 内存不足

**解决方案**：
- 减少 batch size（在 config.json 中）
- 减少时间步数（修改 dispatch.py 中的 `time_steps`）
- 关闭图形界面运行（服务器模式）

### 问题 5: 模型找不到

**症状**：`Model not found: models/xxx.joblib`

**解决方案**：
```bash
# 先训练模型
python src/train.py

# 确认 models/ 目录存在且有文件
ls models/
```

---

## 技术栈

### 机器学习模块

- **LightGBM**: 梯度提升树框架
- **Pandas**: 数据处理
- **NumPy**: 数值计算
- **scikit-learn**: 数据分割、评估指标
- **Matplotlib**: 可视化
- **Joblib**: 模型序列化

### 电力系统模块

- **pandapower**: 电力系统仿真和 OPF
- **NumPy**: 数组运算
- **Pandas**: 结果导出

### 开发工具

- **Python 3.10+**
- **Git**: 版本控制
- **VS Code**: IDE

---

## 高级用法

### 修改发电机成本

编辑 `dispatch.py` 中的 `create_ieee30_base_system()`：

```python
pp.create_poly_cost(
    net, idx, 'gen',
    cp0_eur=0.0,           # 固定成本
    cp1_eur_per_mw=10.0,   # 线性系数 ($/MW)
    cp2_eur_per_mw2=0.0    # 二次系数
)
```

### 添加更多可再生能源

```python
renewable_gens = [
    ('new_solar', bus_number, generation_profile),
    # ... 现有条目
]
```

### 自定义评估指标

在 `utils.py` 中添加新指标：

```python
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_with_more_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}
```

---

## 参考文献

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [pandapower Documentation](https://www.pandapower.org/)
- [IEEE 30-Bus System](https://icseg.iti.illinois.edu/ieee-30-bus-system/)
- [CAISO Data](https://www.caiso.com/)

---

## 许可证

本项目仅供教育和研究使用。

---

## 联系方式

如有问题或建议，请参考项目文档或提交 Issue。

---

**最后更新**: 2026-05-06
