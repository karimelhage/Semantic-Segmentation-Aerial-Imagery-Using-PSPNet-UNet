import matplotlib.pyplot as plt


def plot_losses(total_train_losses, total_val_losses):
    plt.figure(figsize=(10, 8))
    plt.plot(list(range(len(total_train_losses) + 1))[1:120], total_train_losses[:120])
    plt.plot(list(range(len(total_train_losses) + 1))[1:120], total_val_losses[:120])
    # plt.plot(list(range(len(total_train_losses)+1))[1:120], total_val_f1[:120])
    plt.legend(['train loss', 'val loss', 'val f1'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()