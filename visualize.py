import matplotlib.pyplot as plt


def plot_history(hist):
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(hist.epoch, hist.history['loss'], '.-')
    axs[0, 0].set_title('Total Loss')

    axs[0, 1].plot(hist.epoch, hist.history['output_1_loss'], '.-')  # 右上角
    axs[0, 1].set_title('Loss1')

    axs[1, 0].plot(hist.epoch, hist.history['output_2_loss'], '.-')  # 左下角
    axs[1, 0].set_title('Loss2')

    axs[1, 1].plot(hist.epoch, hist.history['output_3_loss'], '.-')  # 右下角
    axs[1, 1].set_title('Loss3')

    fig.suptitle('Loss')
    plt.tight_layout()
    plt.show()
