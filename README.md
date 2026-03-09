# Cobb Angle Prediction（小样本 11 例）

这个仓库提供一个**适合超小样本（例如 11 位患者）**的建模流程，用于回答：

> 新患者输入生物学特征 + 初始 Cobb 角后，若目标是达到某个 Cobb 角，需要施加多少力？

## 方法思路（核心）

直接做“特征 -> 力值”的黑箱回归在 11 例样本上通常不稳定。这里采用两阶段方法：

1. **患者内曲线拟合（力 -> Cobb）**
   - 对每个患者拟合单调衰减曲线：
   - \( Cobb(F)=C_{\infty} + (C_0-C_{\infty})e^{-kF} \)
   - 得到每个患者的两个关键参数：
     - `k`：矫正速度/柔顺性（越大表示同样力下矫正更快）
     - `c_inf`：理论最低 Cobb（矫正极限）

2. **患者间映射（生物学特征 -> 曲线参数）**
   - 由于同一患者生物学特征不变，所以仅在**患者层级**建模。
   - 使用 `BayesianRidge`（多输出）预测 `k` 与 `c_inf`。

3. **反解目标力值**
   - 给定新患者 `c0`（初始 Cobb）、目标 `target_cobb` 以及预测得到的 `k/c_inf`，可解析反解：
   - \( F = -\ln\left(\frac{target-c_{\infty}}{c_0-c_{\infty}}\right)/k \)

## 为什么这个方法适合你当前数据

- 11 例患者太少，端到端 ML 容易过拟合。
- 把问题拆成“**生物机制可解释的中间变量**（`k`、`c_inf`）”再预测，鲁棒性更好。
- 若训练效果一般，也能通过 `k` 和 `F50=ln(2)/k` 作为**中间变量**开展临床分析（谁更“硬”、谁对力更敏感）。

## 输入数据格式

CSV（长表），每行是一帧/一次力值对应观测，至少包含：

- `patient_id`：患者 ID
- `force`：当前施加力值
- `cobb`：该力值下 Cobb 角
- 若干生物学特征列（例如 age、sex、Risser、BMI、柔韧性评分等）

注意：同一患者的生物学特征在多行中应一致。

## 运行方式

```bash
python model_pipeline.py \
  --input your_data.csv \
  --patient-col patient_id \
  --force-col force \
  --cobb-col cobb \
  --target-cobb 25 \
  --outdir outputs
```

如果你希望手动指定可用特征：

```bash
python model_pipeline.py --input your_data.csv --feature-cols age bmi risser flexibility_score
```

## 输出文件

- `outputs/patient_level_parameters.csv`
  - 每位患者拟合得到的 `c0`, `c_inf`, `k`, `f50`, `max_correction_ratio`
- `outputs/features_used.json`
  - 自动识别/手动指定的特征列表
- `outputs/lopo_metrics.json`
  - 留一患者交叉验证（LOPO）指标：`k_mae`, `c_inf_mae`, 及可选 `force_mae_at_target`
- `outputs/deploy_model.json`
  - 可部署参数（标准化参数 + BayesianRidge 系数）

## 建议的下一步

1. 先跑 LOPO，重点看 `force_mae_at_target` 是否在临床可接受区间。
2. 若误差偏大：
   - 增加更有判别力的生物学特征（柔韧性、骨龄、旋转度、肌肉/软组织指标）。
   - 对目标 Cobb 分段建模（如 30°->25°、25°->20°）。
   - 引入层次贝叶斯模型做不确定性建模。
3. 结合中间变量 `f50` 做患者分型（软/中/硬），再在分型内预测力值。

---

如果你愿意，我下一步可以基于你的真实 CSV 列名，直接给你生成一版“可直接跑”的命令和结果解释模板。
