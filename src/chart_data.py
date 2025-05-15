import matplotlib.pyplot as plt
import numpy as np

class ChartData:
    def __init__(self, v_interval: int, master_process=True):
        if not master_process:
            self.update = lambda *args, **kwargs: None
            return
        
        self.fig, (self.ax_loss, self.ax_acc, self.ax_acc_unique_hits) = plt.subplots(3, 1)

        self.v_interval = v_interval

        self.t_loss = np.array([])
        self.t_acc = np.array([])
        self.t_acc_unique_hits = np.array([])
        self.v_loss = np.array([])
        self.v_acc = np.array([])
        self.v_acc_unique_hits = np.array([])

        self.ax_loss.set_title("Loss")
        self.ax_loss.set_xlabel("training batch")
        self.ax_loss.set_ylabel("cat x-entropy loss")

        self.ax_acc.set_title("Accuracy")
        self.ax_acc.set_xlabel("training batch")
        self.ax_acc.set_ylabel("accuracy")

        self.ax_acc_unique_hits.set_title("Unique accuracy hits")
        self.ax_acc_unique_hits.set_xlabel("training batch")
        self.ax_acc_unique_hits.set_ylabel("n unique correct tokens")

        self.ln_t_loss, self.ln_v_loss = self.ax_loss.plot([], self.t_loss, 'b-', [], [], 'r--')
        self.ln_t_acc, self.ln_v_acc = self.ax_acc.plot([], [], 'b-', [], [], 'r--')
        self.ln_t_acc_unique_hits, self.ln_v_acc_unique_hits = self.ax_acc_unique_hits.plot([], [], 'b-', [], [], 'r--')

    def plot(self, _i):
        t_idx = np.arange(len(self.t_loss))
        v_idx = np.arange(len(self.v_loss)) * self.v_interval

        self.ln_t_loss.set_data(t_idx, self.t_loss)
        self.ln_v_loss.set_data(v_idx, self.v_loss)

        self.ln_t_acc.set_data(t_idx, self.t_acc)
        self.ln_v_acc.set_data(v_idx, self.v_acc)

        self.ln_t_acc_unique_hits.set_data(t_idx, self.t_acc_unique_hits)
        self.ln_v_acc_unique_hits.set_data(v_idx, self.v_acc_unique_hits)

        self.ax_loss.relim()
        self.ax_loss.autoscale()
        self.ax_acc.relim()
        self.ax_acc.autoscale()
        self.ax_acc_unique_hits.relim()
        self.ax_acc_unique_hits.autoscale()

    def update(self, t_loss=[], t_acc=[], t_acc_unique_hits=[], v_loss=[], v_acc=[], v_acc_unique_hits=[]):
        self.t_loss = np.append(self.t_loss, t_loss)
        self.v_loss = np.append(self.v_loss, v_loss)

        self.t_acc = np.append(self.t_acc, t_acc)
        self.v_acc = np.append(self.v_acc, v_acc)

        self.t_acc_unique_hits = np.append(self.t_acc_unique_hits, t_acc_unique_hits)
        self.v_acc_unique_hits = np.append(self.v_acc_unique_hits, v_acc_unique_hits)

    def save_plot(self):
        try:
            plot_fname = "last_plot.png"
            print(f"Saving plot to {plot_fname}...")

            self.fig.savefig(plot_fname)
            plt.close(self.fig)

        except Exception as e:
            print(f"Could not save plot:\n{e}")
