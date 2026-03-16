import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train Motif-Based Model for Molecular Generation")

    # 数据相关参数
    parser.add_argument('--dataset', type=str, default='MUV', help='Dataset name (e.g., ZINC250K)')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Directory containing the dataset')

    # 模型相关参数
    parser.add_argument('--node_feature_dim', type=int, default=9, help='Dimension of node features')
    parser.add_argument('--edge_feature_dim', type=int, default=3, help='Dimension of edge features')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension of the model')

    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default =64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=5, help='Patience for learning rate scheduler')
    parser.add_argument('--factor', type=float, default=0.6, help='Factor for reducing learning rate')
    parser.add_argument('--loss_fn', type=str, default='huber', choices=['mse', 'huber'], help='Loss function to use')
    parser.add_argument('--alpha', type=float, default=0.6, help='contrastive learning rate ring')
    parser.add_argument('--beta', type=float, default=0.4, help='contrastive learning rate noring')
    parser.add_argument('--Pair_MLP', type=bool, default=True, help='是否用contrastive learning')
    parser.add_argument('--is_contrastive', type=bool, default=True, help='是否用contrastive learning')
    parser.add_argument('--use_Guide', type=bool, default=True, help='是否使用Guide')
    parser.add_argument('--use_gating', type=bool, default=True, help='是否使用use_gating')
    parser.add_argument('--use_head_interaction', type=bool, default=True, help='是否使用use_head_interaction')
    parser.add_argument('--label_thresh_ratio', type=float, default=0.9, help='计算分子之间距离')

    # 保存相关参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints/', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='Directory to save training logs')

    # 设备相关参数
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to run the model on')

    return parser.parse_args()