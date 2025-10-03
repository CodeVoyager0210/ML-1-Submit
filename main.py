#!/usr/bin/env python3
"""
加州房价回归实验主程序

本程序是加州房价回归实验的入口点，负责按顺序执行各个实验阶段：
1. 数据准备 (Data Preparation)
2. 数据预处理 (Data Preprocessing)
3. 模型搭建 (Model Building)
4. 模型训练 (Model Training)
5. 模型优化 (Model Optimization)
6. 模型评估 (Model Evaluation)

作者: Claude
日期: 2025-01-01
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.data_loader import HousingDataLoader, quick_data_overview
from src.utils.visualization import HousingDataVisualizer
from src.data_preparation import DataPreparationPipeline


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    设置日志配置

    Args:
        log_level (str): 日志级别
        log_file (Optional[str]): 日志文件路径
    """
    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # 清除已有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 添加文件处理器（如果指定）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数

    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="加州房价回归实验主程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行完整的数据准备阶段
  python main.py --phase preparation --data-dir data --output-dir results

  # 仅运行数据质量检查
  python main.py --phase preparation --data-dir data --output-dir results --skip-plots

  # 运行所有阶段（完整实验）
  python main.py --phase all --data-dir data --output-dir results

  # 使用自定义k值进行交叉验证
  python main.py --phase all --data-dir data --output-dir results --k-folds 10

  # 启用详细日志
  python main.py --phase preparation --data-dir data --output-dir results --log-level DEBUG
        """
    )

    parser.add_argument(
        "--phase",
        type=str,
        choices=["preparation", "preprocessing", "building", "training", "optimization", "evaluation", "all"],
        default="preparation",
        help="要运行的实验阶段 (默认: preparation)"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="数据目录路径 (默认: data)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="输出目录路径 (默认: results)"
    )

    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        help="K折交叉验证的折数 (默认: 5)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="日志文件路径 (可选)"
    )

    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="跳过可视化图表生成"
    )

    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="强制覆盖已存在的输出文件"
    )

    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="测试模式：使用少量数据进行快速验证"
    )

    return parser.parse_args()


def run_data_preparation(args: argparse.Namespace) -> None:
    """
    运行数据准备阶段

    Args:
        args (argparse.Namespace): 命令行参数
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("开始阶段一：数据准备 (Data Preparation)")
    logger.info("=" * 60)

    try:
        # 创建数据准备管道
        pipeline = DataPreparationPipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )

        # 步骤1：加载和验证数据
        logger.info("步骤 1/3: 加载和验证数据")
        if args.test_mode:
            # 测试模式：只加载前1000行数据
            pipeline.data_loader.data = pipeline.data_loader.load_raw_data().head(1000)
            logger.info("测试模式：仅使用前1000行数据")
        else:
            pipeline.load_and_validate_data()

        # 步骤2：探索性数据分析（除非跳过图表）
        if not args.skip_plots:
            logger.info("步骤 2/3: 探索性数据分析")
            pipeline.perform_eda()
        else:
            logger.info("步骤 2/3: 跳过探索性数据分析（--skip-plots）")

        # 步骤3：运行完整管道
        logger.info("步骤 3/3: 运行完整数据准备管道")
        pipeline.run_complete_pipeline()

        logger.info("✅ 数据准备阶段完成！")
        logger.info(f"📊 结果已保存到: {args.output_dir}")

    except Exception as e:
        logger.error(f"❌ 数据准备阶段失败: {str(e)}")
        raise


def run_data_preprocessing(args: argparse.Namespace) -> None:
    """
    运行数据预处理阶段

    Args:
        args (argparse.Namespace): 命令行参数
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("开始阶段二：数据预处理 (Data Preprocessing)")
    logger.info("=" * 60)

    logger.info("🚧 数据预处理阶段待实现")
    logger.info("将包含以下步骤:")
    logger.info("1. 缺失值处理")
    logger.info("2. 异常值检测与处理")
    logger.info("3. 分类变量编码")
    logger.info("4. 特征缩放和标准化")
    logger.info("5. 特征工程")
    logger.info("6. 数据集划分（K折交叉验证）")


def run_model_building(args: argparse.Namespace) -> None:
    """
    运行模型搭建阶段

    Args:
        args (argparse.Namespace): 命令行参数
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("开始阶段三：模型搭建 (Model Building)")
    logger.info("=" * 60)

    logger.info("🚧 模型搭建阶段待实现")
    logger.info("将包含以下步骤:")
    logger.info("1. 线性回归模型")
    logger.info("2. 正则化回归模型（Ridge/Lasso/Elastic Net）")
    logger.info("3. 模型配置和超参数设置")


def run_model_training(args: argparse.Namespace) -> None:
    """
    运行模型训练阶段

    Args:
        args (argparse.Namespace): 命令行参数
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("开始阶段四：模型训练 (Model Training)")
    logger.info("=" * 60)

    logger.info("🚧 模型训练阶段待实现")
    logger.info("将包含以下步骤:")
    logger.info("1. 模型训练流程")
    logger.info("2. 交叉验证实施")
    logger.info("3. 训练过程监控")
    logger.info("4. 初步性能评估")


def run_model_optimization(args: argparse.Namespace) -> None:
    """
    运行模型优化阶段

    Args:
        args (argparse.Namespace): 命令行参数
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("开始阶段五：模型优化 (Model Optimization)")
    logger.info("=" * 60)

    logger.info("🚧 模型优化阶段待实现")
    logger.info("将包含以下步骤:")
    logger.info("1. 超参数调优")
    logger.info("2. 模型集成")
    logger.info("3. 特征选择优化")
    logger.info("4. 模型性能比较")


def run_model_evaluation(args: argparse.Namespace) -> None:
    """
    运行模型评估阶段

    Args:
        args (argparse.Namespace): 命令行参数
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("开始阶段六：模型评估 (Model Evaluation)")
    logger.info("=" * 60)

    logger.info("🚧 模型评估阶段待实现")
    logger.info("将包含以下步骤:")
    logger.info("1. 最终模型性能评估")
    logger.info("2. 特征重要性分析")
    logger.info("3. 模型解释性分析")
    logger.info("4. 实验总结报告")


def quick_data_check(data_dir: str) -> None:
    """
    快速数据检查

    Args:
        data_dir (str): 数据目录路径
    """
    logger = logging.getLogger(__name__)
    logger.info("🔍 执行快速数据检查...")

    try:
        # 获取数据概览
        overview = quick_data_overview(data_dir)

        # 显示基本信息
        basic_info = overview['basic_info']
        logger.info(f"数据形状: {basic_info['shape']}")
        logger.info(f"特征数量: {len(basic_info['columns'])}")
        logger.info(f"总缺失值: {basic_info['total_nulls']}")
        logger.info(f"内存使用: {basic_info['memory_usage'] / 1024 / 1024:.2f} MB")

        # 显示目标变量信息
        target_info = overview['target_info']
        logger.info(f"目标变量范围: {target_info['statistics']['min']:.2f} - {target_info['statistics']['max']:.2f}")
        logger.info(f"目标变量均值: {target_info['statistics']['mean']:.2f}")

        # 显示数据质量信息
        quality_report = overview['quality_report']
        logger.info(f"重复数据: {quality_report['duplicates']['total_duplicates']}")

        logger.info("✅ 数据检查完成")

    except Exception as e:
        logger.error(f"❌ 数据检查失败: {str(e)}")
        raise


def main() -> None:
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()

    # 设置日志
    log_file = args.log_file or f"{args.output_dir}/experiment.log"
    setup_logging(args.log_level, log_file)

    logger = logging.getLogger(__name__)
    logger.info("🚀 启动加州房价回归实验")
    logger.info(f"实验阶段: {args.phase}")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"K折交叉验证: {args.k_folds}")

    try:
        # 创建输出目录
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # 执行快速数据检查
        if Path(args.data_dir).exists():
            quick_data_check(args.data_dir)
        else:
            logger.warning(f"数据目录不存在: {args.data_dir}")

        # 根据指定阶段执行相应功能
        if args.phase == "preparation":
            run_data_preparation(args)
        elif args.phase == "preprocessing":
            run_data_preprocessing(args)
        elif args.phase == "building":
            run_model_building(args)
        elif args.phase == "training":
            run_model_training(args)
        elif args.phase == "optimization":
            run_model_optimization(args)
        elif args.phase == "evaluation":
            run_model_evaluation(args)
        elif args.phase == "all":
            run_data_preparation(args)
            run_data_preprocessing(args)
            run_model_building(args)
            run_model_training(args)
            run_model_optimization(args)
            run_model_evaluation(args)

        logger.info("🎉 实验完成！")

    except KeyboardInterrupt:
        logger.info("⚠️ 实验被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 实验失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()