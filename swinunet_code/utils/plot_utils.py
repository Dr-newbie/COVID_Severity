import matplotlib.pyplot as plt


def plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, args):
    """
    학습 및 검증 손실과 정확도를 플로팅하고 저장하는 함수.

    Args:
        train_losses (list): 학습 손실 값 리스트.
        val_losses (list): 검증 손실 값 리스트.
        train_accuracies (list): 학습 정확도 값 리스트.
        val_accuracies (list): 검증 정확도 값 리스트.
        args: Argument parser로 전달된 설정 값.

    Returns:
        None
    """
    args_dict = vars(args)
    args_str = '_'.join([f'{k}_{v}' for k, v in args_dict.items() if k in ['batch_size', 'learning_rate', 'epochs']])

    # 학습 및 검증 손실 플롯
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    loss_plot_path = f'loss_plot_{args_str}.png'
    plt.savefig(loss_plot_path)
    plt.close()

    # 학습 및 검증 정확도 플롯
    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    accuracy_plot_path = f'accuracy_plot_{args_str}.png'
    plt.savefig(accuracy_plot_path)
    plt.close()

    print(f"Saved plots: {loss_plot_path}, {accuracy_plot_path}")
