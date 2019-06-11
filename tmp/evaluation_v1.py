from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import powerlaw

from collections import Counter

title_fontsize = 22
label_fontsize = 18
tick_fontsize = 16
legend_fontsize = 18
figsize = (8, 6)
linewidth = 2.0
n_bins = 0


class Evaluation(object):

    def __init__(self, network=None, attributes=None):
        self.network = network
        self.attributes = attributes
        #self.degree_sequence = sorted(dict(nx.degree(graph)).values(), reverse=True)  # degree sequence

    def draw_pdf(self, sequence, fig_ax=None, title=None, legend_label=None,
                 x_label="Counts", y_label="p(X)", style='b-', marker='o'):
        if not fig_ax:
            fig_ax = plt.subplots(figsize=figsize)
        fit = powerlaw.Fit(sequence, xmin=1)
        alhpa = fit.power_law.alpha
        xmin = fit.xmin
        dis = fit.power_law.D
        mu = fit.lognormal.mu
        sigma = fit.lognormal.sigma
        R, p = fit.distribution_compare('power_law', 'lognormal')
        print(dis, R, p)
        powerlaw.plot_pdf(sequence, linewidth=linewidth, marker=marker,
                          color=style[0], ax=fig_ax[1],
                          label=r"%s ($\mu$=%.2f, $\sigma$=%.2f)" % (legend_label, mu, sigma))
                          #label=r"%s ($\alpha$=%.2f)" % (legend_label, alhpa))
                          #label=r"%s ($\alpha$=%.2f, p=%.2f)" % (legend_label, alhpa, p))
        fit.lognormal.plot_pdf(color=style[0], linestyle='--', ax=fig_ax[1])
        #fit.power_law.plot_pdf(color=style[0], linestyle='--', ax=fig_ax[1])
        fig_ax[1].set_xlabel(x_label, fontsize=label_fontsize)
        fig_ax[1].set_ylabel(y_label, fontsize=label_fontsize)
        fig_ax[1].tick_params(size=tick_fontsize)
        if legend_label:
            fig_ax[1].legend(fontsize=legend_fontsize)
        if title:
            fig_ax[1].set_title(title, fontsize=title_fontsize)
        fig_ax[0].tight_layout()
        return fig_ax

    def draw_cdf(self, sequence, fig_ax=None, title=None, legend_label=None,
                 x_label="Degree", y_label="CDF", style='b-', marker='o', is_log=True):
        if not fig_ax:
            fig_ax = plt.subplots(figsize=figsize)
        powerlaw.plot_cdf(sequence, linewidth=linewidth, marker=marker,
                     color=style[0], ax=fig_ax[1],
                     label=legend_label)
        fig_ax[1].grid(True)
        if title:
            fig_ax[1].set_title(title, fontsize=title_fontsize)
        fig_ax[1].set_xlabel(x_label, fontsize=label_fontsize)
        fig_ax[1].set_ylabel(y_label, fontsize=label_fontsize)
        #fig_ax[1].set_xlim(left=0, right=500)
        plt.yscale('linear')
        if legend_label:
            fig_ax[1].legend(fontsize=legend_fontsize)
        #fig_ax[1].legend(fontsize=legend_fontsize, loc='lower right')
        fig_ax[1].tick_params(size=tick_fontsize)
        fig_ax[0].tight_layout()
        return fig_ax

    def draw_pdf_copy(self, sequence, fig_ax=None, title=None, legend_label=None,
                 x_label="Rank(%)", y_label="Degree", style='b-', marker='o'):
        if not fig_ax:
            fig_ax = plt.subplots(figsize=figsize)# print("Degree sequence", degree_sequence)
        sequence.sort(reverse=True)
        fig_ax[1].loglog(np.arange(1, len(sequence)+1)/(len(sequence))*100,
                         sequence, style, marker=marker, label=legend_label, linewidth=linewidth)
        fig_ax[1].set_xlabel(x_label, fontsize=label_fontsize)
        fig_ax[1].set_ylabel(y_label, fontsize=label_fontsize)
        fig_ax[1].tick_params(size=tick_fontsize)
        if legend_label:
            fig_ax[1].legend(fontsize=legend_fontsize)
        if title:
            fig_ax[1].set_title(title, fontsize=title_fontsize)
        fig_ax[0].tight_layout()
        return fig_ax

    def draw_pdf2(self, sequence, fig_ax=None, title=None, legend_label=None,
                 x_label="Video Duration (s)", y_label="Percentage", style='b-', marker=None):
        if not fig_ax:
            fig_ax = plt.subplots(figsize=figsize)# print("Degree sequence", degree_sequence)
        n_bins = 1000#max(sequence)
        counts, bin_edges = np.histogram(sequence, bins=np.arange(n_bins))
        counts = counts/counts.sum()
        fig_ax[1].plot(bin_edges[1:], counts,
                         style, label=legend_label, linewidth=linewidth)
        fig_ax[1].set_xlabel(x_label, fontsize=label_fontsize)
        fig_ax[1].set_ylabel(y_label, fontsize=label_fontsize)
        fig_ax[1].tick_params(size=tick_fontsize)
        #plt.xscale('log')
        if legend_label:
            fig_ax[1].legend(fontsize=legend_fontsize)
        if title:
            fig_ax[1].set_title(title, fontsize=title_fontsize)
        fig_ax[0].tight_layout()
        return fig_ax

    def draw_cdf_copy(self, sequence, fig_ax=None, title=None, legend_label=None,
                 x_label="Degree", y_label="CDF", style='b-', marker='o', is_log=True):
        if not fig_ax:
            fig_ax = plt.subplots(figsize=figsize)
        global n_bins
        if n_bins == 0:
            n_bins = max(sequence)
        counts, bin_edges = np.histogram(sequence, bins=np.arange(n_bins), density=True)
        cdf = np.cumsum(counts)
        n, index = self.cdf_n_percent(cdf, 0.8)
        print("%.2f" % n, bin_edges[1:][index])
        fig_ax[1].plot(bin_edges[1:], cdf, style, marker=marker, label=legend_label, linewidth=linewidth)
        fig_ax[1].grid(True)
        if title:
            fig_ax[1].set_title(title, fontsize=title_fontsize)
        fig_ax[1].set_xlabel(x_label, fontsize=label_fontsize)
        fig_ax[1].set_ylabel(y_label, fontsize=label_fontsize)
        #fig_ax[1].set_xlim(left=0, right=500)
        if is_log:
            plt.xscale('log')
        if legend_label:
            fig_ax[1].legend(fontsize=legend_fontsize)
        #fig_ax[1].legend(fontsize=legend_fontsize, loc='lower right')
        fig_ax[1].tick_params(size=tick_fontsize)
        fig_ax[0].tight_layout()
        return fig_ax

    def draw_unique_samples(self, unique_samples, n_iterations):
        # for three bars
        fig, ax = plt.subplots(figsize=figsize)
        bar_width = 0.25
        opacity = 0.4
        index = np.arange(len(n_iterations))
        colors = ['r', 'g', 'b']
        for i,method in enumerate(unique_samples):
            ax.bar(index + bar_width*(i-1), unique_samples[method], bar_width,
                   alpha=opacity, color=colors[i], label=method)
            for x, y in zip(index + bar_width*(i-1), unique_samples[method]):
                ax.text(x + 0.05, y + 0.05, '%d' % y, ha='center', va='bottom')
        ax.set_xlabel('Iterations', fontsize=label_fontsize)
        ax.set_ylabel('# of Unique Samples', fontsize=label_fontsize)
        ax.tick_params(size=tick_fontsize)
        #ax.set_title('Unique Samples of Different Iterations', fontsize=title_fontsize)
        ax.set_xticks(index)
        ax.set_xticklabels(n_iterations)
        ax.legend(fontsize=legend_fontsize)

        fig.tight_layout()
        return fig, ax

    def draw_realation(self, x, y, fig_ax=None, title=None,
                       x_label="# of followers", y_label="# of followings",
                       style='b-', marker='o'):
        if not fig_ax:
            fig_ax = plt.subplots(figsize=figsize)# print("Degree sequence", degree_sequence)

        fig_ax[1].scatter(x, y)#, style, marker=marker, linewidth=linewidth)
        fig_ax[1].set_xlabel(x_label, fontsize=label_fontsize)
        fig_ax[1].set_ylabel(y_label, fontsize=label_fontsize)
        fig_ax[1].tick_params(size=tick_fontsize)
        fig_ax[1].set_ylim(bottom=1, top=5000)
        fig_ax[1].set_xlim(left=1, right=40000)
        fig_ax[1].set_yscale('log', basey=2)
        fig_ax[1].set_xscale('log', basex=2)

        if title:
            fig_ax[1].set_title(title, fontsize=title_fontsize)
        fig_ax[0].tight_layout()
        return fig_ax

    def draw_pie(self, sequence):
        fig_ax = plt.subplots(figsize=figsize)  # print("Degree sequence", degree_sequence)
        c = Counter(sequence)
        x = np.array(list(c.keys()))
        y = np.array(list(c.values()))
        percent = 100. * y / y.sum()
        patches, texts = fig_ax[1].pie(y, startangle=90, radius=1.2)
        labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(x, percent)]
        patches, labels, dummy = zip(*sorted(zip(patches, labels, y),
                                             key=lambda x: x[2],
                                             reverse=True))
        fig_ax[1].legend(patches, labels, loc='center left', bbox_to_anchor=(-0.4, 1.),
                         fontsize=legend_fontsize)
        return fig_ax

    def draw_pie2(self, sequence):
        fig, ax = plt.subplots(figsize=figsize)  # print("Degree sequence", degree_sequence)
        c = Counter(sequence)
        labels = list(c.keys())
        data = list(c.values())
        wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
                  bbox=bbox_props, zorder=0, va="center")
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1) / 2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(labels[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                        horizontalalignment=horizontalalignment, **kw)
        return fig, ax


    def cdf_n_percent(self, cdf, n):
        i = 0
        while i < len(cdf):
            if i == len(cdf) - 1:
                break
            if cdf[i]<=n and cdf[i+1]>n:
                break
            i += 1
        return n, i




if __name__ == '__main__':
    eval = Evaluation()
    fig, ax = eval.draw_unique_samples({'a': [1, 2, 3], 'b': [2, 2, 2],
                                        'c': [3, 3, 3]}, [100, 200, 300])
    plt.show()
    '''
    # plot with networkx
    G = nx.gnp_random_graph(100, 0.02)
    plt = eval.draw_pdf(option='new',G=G)
    plt.show()
    G = nx.gnp_random_graph(100, 0.02)
    plt = eval.draw_cdf(option='new', G=G)
    plt.show()
    '''
    '''
    from sampling import *
    from util import *

    file_path = r'G:\Sampling\dataset\Cit-HepPh.txt'
    graph = load_graph(file_path)
    original_degrees = [len(graph[node]) for node in graph]

    instance = MHRW(graph=graph,num=100)
    start = list(graph.keys())
    root = random.choice(start)
    instance.run(root)
    degrees = instance.get_degrees()
    degrees_duplicate = instance.get_degrees_duplicate()
    print(np.median(degrees))
    print(np.median(degrees_duplicate))
    # pdf plot
    fig_ax = eval.draw_pdf(degrees=original_degrees, label='Original', style='b-', marker='o')
    fig_ax = eval.draw_pdf(fig_ax=fig_ax, degrees=degrees_duplicate, label='MHRW', style='r-.', marker='<')
    fig_ax[0].savefig('./test/pdf.png')
    #cdf plot
    fig_ax = eval.draw_cdf(degrees=original_degrees, label='Original', style='b-', marker='o')
    fig_ax = eval.draw_cdf(fig_ax=fig_ax, degrees=degrees_duplicate, label='MHRW', style='r-.', marker='<')
    fig_ax[0].savefig('./test/cdf.png')
    '''
