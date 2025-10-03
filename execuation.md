# 激活虚拟环境
conda activate ML-1

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 运行阶段一数据准备阶段
python main.py --phase preparation --data-dir data --output-dir results

# 阶段一测试模式（快速验证）
python main.py --phase preparation --data-dir data --output-dir results --test-mode

# 运行阶段二数据预处理阶段
python main.py --phase preprocessing --data-dir data --output-dir results

# 运行阶段三模型搭建阶段（完整评估）
python src/model_building.py --data-dir split_data --models-dir models --results-dir results

# 阶段三快速测试模式（单折评估）
python src/model_building.py --data-dir split_data --models-dir models --results-dir results --test-mode

# 运行阶段四模型训练阶段（完整训练）
python src/model_training.py --data-dir split_data --models-dir models --results-dir results

# 阶段四快速测试模式（单折训练）
python src/model_training.py --data-dir split_data --models-dir models --results-dir results --test-mode

# 阶段四想要日志记录，使用下main的日志记录增强模式：
python src/model_training_with_logging.py --data-dir split_data --models-dir models --results-dir results 时：

  - 文件日志: 所有日志信息会同时输出到控制台和文件
  - 日志文件位置: results/logs/model_training_YYYYMMDD_HHMMSS.log
  - 文件命名: 带时间戳，每次运行都会生成新的日志文件



# 本回归任务中，以下几个库可以手动实现而不需要调用库：

  1. 线性模型系列

  线性回归 (Linear Regression)

  可以手动实现：
  - 最小二乘法的解析解：(X^T X)^(-1) X^T y
  - 梯度下降法求解参数

  岭回归 (Ridge Regression)

  可以手动实现：
  - 解析解：(X^T X + λI)^(-1) X^T y
  - 梯度下降法（加入L2正则化）

  Lasso回归

  可以手动实现：
  - 坐标下降法（Coordinate Descent）
  - 近端梯度下降法（Proximal Gradient Descent）

  Elastic Net

  可以手动实现：
  - 结合L1和L2正则化的优化算法

  2. 集成学习方法

  随机森林 (Random Forest)

  可以手动实现：
  - 决策树的构建（CART算法）
  - Bootstrap采样
  - 特征随机选择
  - 多树投票机制

  梯度提升 (Gradient Boosting)

  可以手动实现：
  - 梯度计算和残差拟合
  - 学习率调整
  - 序列模型构建

  3. 神经网络

  MLP (多层感知机)

  可以手动实现：
  - 前向传播
  - 反向传播算法
  - 各种激活函数
  - 优化器（SGD、Adam等）

  4. 基础工具库

  评估指标计算

  可以手动实现：
  - MSE：mean((y_true - y_pred)^2)
  - MAE：mean(|y_true - y_pred|)
  - R²：1 - sum((y_true - y_pred)^2) / sum((y_true - y_mean)^2)
  - RMSE：sqrt(MSE)

  数据预处理

  可以手动实现：
  - 标准化：(x - mean) / std
  - 归一化：(x - min) / (max - min)
  - 训练集/验证集划分

  手动实现的价值

  1. 深入理解算法原理：通过手动实现，可以更好地理解算法的数学原理和实现细节
  2. 面试准备：很多科技公司面试会要求手写算法
  3. 定制化需求：可以根据具体问题定制算法细节
  4. 性能优化：可以针对特定数据集优化算法性能

  建议的实现顺序

  1. 线性回归（最基础，理解最小二乘法）
  2. 岭回归（理解正则化的作用）
  3. 梯度下降优化器（理解优化算法）
  4. 决策树（理解树结构算法）
  5. 随机森林（理解集成学习）
  6. 简单的神经网络（理解深度学习基础）

# 新增超参数：
- **k_fold_selectable**：允许用户选择交叉验证折数（默认5折）
- **use_library_models**：二进制参数（1=使用sklearn库，0=使用手动实现）
- **early_stopping_patience**：早停机制的耐心参数
- **regularization_strength_range**：正则化强度范围
- **learning_rate_schedule**：学习率调度策略

###  阶段五执行

  - 快速测试：使用 --test-mode --k-fold 2
  - 完整优化：使用 --k-fold 5（默认）
  - 交互式调参：使用 --interactive
  - 仅手动实现：使用 --manual-only（节省时间）
  - 仅库模型：使用 --use-library（更稳定）

# 使用手动实现模型进行超参数优化
 1. 首先运行手动实现模型训练：
  python src/manual_model_training.py --data-dir split_data --models-dir models --results-dir results
  # 可以在results/logs目录下查看对应时间戳的详细日志
2. 然后运行模型优化：
  python src/model_optimization.py --data-dir split_data --models-dir models --results-dir results --k-fold 5 --manual-only
  # 为了节约优化时间，使用创建的选择性优化脚本 - 只优化线性模型
  python src/fast_optimization.py --data-dir split_data --models-dir models --results-dir results --k-fold 5
  1. 线性模型已经证明性能良好（R² ≈ 0.65）
  2. 可以快速获得优化结果
  3. 树模型性能很差（R² ≈ 0.003），优化价值不大


  # 使用sklearn库模型进行超参数优化
  python src/model_optimization.py --data-dir split_data --models-dir models --results-dir results --k-fold 5 --use-library

  # 同时使用手动实现和库模型进行对比
  python src/model_optimization.py --data-dir split_data --models-dir models --results-dir results --k-fold 5

  # 测试模式（快速运行，仅少量模型）
  python src/model_optimization.py --data-dir split_data --models-dir models --results-dir results --k-fold 2 --test-mode

  交互式可视化模式

  # 启动交互式界面（推荐用于参数调优实验）
  python src/model_optimization.py --data-dir split_data --models-dir models --results-dir results --interactive

  # 学习率是参数更新的 “步长”。它决定了模型参数向 “损失函数最小值” 移动的幅度
  与 “固定学习率” 不同，动态学习率会在训练过程中自动调整步长，常见策略：
  学习率衰减：随着训练轮数增加，学习率逐渐减小（如每过 100 轮，学习率乘以 0.9）。初期用大学习率快速接近最优区域，后期用小学习率精细搜索；
  基于验证集性能调整：若验证集损失长时间停止下降，自动降低学习率（如变为原来的 1/10），让模型更精准收敛。
  动态学习率平衡了 “训练速度” 与 “收敛精度”，避免固定学习率的局限性。
  # 线性模型最优超参数推荐

  1. Ridge Regression (岭回归)

  - 正则化强度 (alpha): 0.001
  - 验证R²: 0.6530
  - 特点: 较小的正则化强度，说明数据不太需要强的正则化约束

  2. Lasso Regression (套索回归)

  - 正则化强度 (alpha): 10.0
  - 验证R²: 0.6530
  - 特点: 较大的正则化强度，有助于特征选择

  3. Elastic Net (弹性网络)

  - 正则化强度 (alpha): 使用默认值（结果中显示为空）
  - L1比例 (l1_ratio): 使用默认值0.5
  - 验证R²: 0.6184
  - 特点: 性能略低于其他线性模型

  参数分析

  1. Ridge Regression的alpha=0.001：
    - 很小的正则化强度，说明模型倾向于使用更复杂的拟合
    - 适合处理多重共线性问题
  2. Lasso Regression的alpha=10.0：
    - 较大的正则化强度，会进行特征选择
    - 适合需要稀疏解的场景
  3. 关于学习率：
    - 线性模型（Linear、Ridge、Lasso）通常使用解析解，不需要学习率
    - 如果使用梯度下降法求解，建议学习率范围：0.001-0.1

  实际建议

  最佳模型选择：
  - 如果需要特征选择：使用 Lasso (alpha=10.0)
  - 如果需要稳定性：使用 Ridge (alpha=0.001)
  - 如果平衡性能：使用 Linear Regression（无正则化）

  学习率设置（仅当使用梯度下降时）：
  - 初始学习率：0.01
  - 学习率调度：可使用指数衰减或余弦退火
  - 批量大小：32-128之间