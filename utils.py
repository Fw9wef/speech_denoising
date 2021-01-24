from settings import path_to_log, path_to_fig
import matplotlib.pyplot as plt


class Log(path_to_log=path_to_log, path_to_fig=path_to_fig):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.path_to_log = path_to_log
        self.path_to_plot = path_to_fig

    def write_log(self, epoch_n, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        message = "epoch number : %d\ttrain loss : %.6f\tval loss : %.6f" % (epoch_n, train_loss, val_loss)
        print(message)
        with open(self.path_to_log, 'a') as f:
            f.write(message+str('\n'))

    def save_plots(self):
        epochs = [i for i in range(1, len(self.train_losses))]
        plt.figure(figsize=(10, 10))
        plt.plot(self.train_losses, epochs, c='r')
        plt.plot(self.val_losses, epochs, c='g')
        plt.savefig(self.path_to_plot)
