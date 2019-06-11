from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import powerlaw_revised

from collections import Counter
#from pymc3.diagnostics import geweke
from util import geweke
from matplotlib import colors

title_fontsize = 24
label_fontsize = 20
tick_fontsize = 18
legend_fontsize = 18
figsize = (8, 6)
linewidth = 2.0
n_bins = 0


class Evaluation(object):

    def __init__(self, network=None, attributes=None):
        self.network = network
        self.attributes = attributes
        #self.degree_sequence = sorted(dict(nx.degree(graph)).values(), reverse=True)  # degree sequence

    def draw_zscore(self, sequence=None, weights=None, fig_ax=None, base_line=False,
                    legend_label=None, x_label="# for users",
                    y_label="Geweke Z-Score", linestyle='-.', color='red'):
        sequence = np.array(sequence)
        if not fig_ax:
            fig_ax = plt.subplots(figsize=figsize)
        if base_line:
            fig_ax[1].plot([0, 100000], [0.01, 0.01], 'b--', linewidth=linewidth)
            fig_ax[1].plot([0, 100000], [-0.01,-0.01],'b--', linewidth=linewidth)
            #fig_ax[1].plot([0, 100000], [0.1, 0.1], 'b--', linewidth=linewidth)
            #fig_ax[1].plot([0, 100000], [-0.1,-0.1],'b--', linewidth=linewidth)
            return fig_ax
        if not weights is None:
            weights = np.array(weights)
        iterations = []
        z_scroes = []
        for i in range(100, len(sequence),100):
            iterations.append(i)
            score = geweke(sequence[:i], weights=weights[:i],intervals=2)[:, 1].mean()
            #score = geweke(sequence[:i], intervals=2)[:, 1].mean()
            z_scroes.append(score)
        fig_ax[1].plot(iterations, z_scroes,linestyle=linestyle, color=color,
                       label=legend_label, linewidth=linewidth)
        fig_ax[1].set_xlabel(x_label, fontsize=label_fontsize)
        fig_ax[1].set_ylabel(y_label, fontsize=label_fontsize)
        if legend_label:
            fig_ax[1].legend(fontsize=legend_fontsize)
        fig_ax[0].tight_layout()
        return fig_ax

    def draw_pdf(self, sequence, weights=None, fit_function=None, fig_ax=None, title=None,
                 legend_label=None, x_label="Counts", y_label="p(X)", style='b-', marker='o',
                 x_scale='log', y_scale='log', xmin=1):
        sequence = np.array(sequence)
        if not weights is None:
            weights = np.array(weights)
        if not fig_ax:
            fig_ax = plt.subplots(figsize=figsize)
        if fit_function == 'power_law':
            fit = powerlaw_revised.Fit(sequence, weights=weights, xmin=xmin)
            alpha = fit.power_law.alpha
            D = fit.power_law.D
            xmin = fit.xmin
            powerlaw_revised.plot_pdf(sequence, weights=weights, linewidth=linewidth, marker=marker,
                                      color=style[0], ax=fig_ax[1], x_scale=x_scale, y_scale=y_scale,
                                      label=r"$\alpha$=%.2f, K-S distance=%.2f" % (alpha, D))
                              #label=r"%s ($\alpha$=%.2f)" % (legend_label, alpha))
            fit.power_law.plot_pdf(color=style[0], linestyle='--', ax=fig_ax[1])
        elif fit_function == 'lognormal':
            fit = powerlaw_revised.Fit(sequence, weights=weights, xmin=xmin)
            mu = fit.lognormal.mu
            sigma = fit.lognormal.sigma
            powerlaw_revised.plot_pdf(sequence, weights=weights, linewidth=linewidth, marker=marker,
                               color=style[0], ax=fig_ax[1], x_scale=x_scale, y_scale=y_scale,
                               label=r"%s ($\mu$=%.2f, $\sigma$=%.2f)" % (legend_label, mu, sigma))
            fit.lognormal.plot_pdf(color=style[0], linestyle='--', ax=fig_ax[1])
        else:
            powerlaw_revised.plot_pdf(sequence, weights=weights, linewidth=linewidth, marker=marker,
                          x_scale=x_scale, y_scale=y_scale, color=style[0], ax=fig_ax[1], label=legend_label)
        fig_ax[1].set_xlabel(x_label, fontsize=label_fontsize)
        fig_ax[1].set_ylabel(y_label, fontsize=label_fontsize)
        fig_ax[1].tick_params(size=tick_fontsize)
        if legend_label:
            fig_ax[1].legend(fontsize=legend_fontsize)
        if title:
            fig_ax[1].set_title(title, fontsize=title_fontsize)
        fig_ax[0].tight_layout()
        return fig_ax

    def draw_cdf(self, sequence, weights=None, fig_ax=None, title=None, legend_label=None,
                 x_label="Degree", y_label="CDF", style='b-', marker='o',
                 x_scale='log', y_scale='log'):
        sequence = np.array(sequence)
        if not weights is None:
            weights = np.array(weights)
        if not fig_ax:
            fig_ax = plt.subplots(figsize=figsize)
        #import powerlaw
        powerlaw_revised.plot_cdf(sequence, weights=weights, linewidth=linewidth, marker=marker,
                      color=style[0], ax=fig_ax[1], label=legend_label,
                      x_scale=x_scale, y_scale=y_scale)
        fig_ax[1].grid(True)
        if title:
            fig_ax[1].set_title(title, fontsize=title_fontsize)
        fig_ax[1].set_xlabel(x_label, fontsize=label_fontsize)
        fig_ax[1].set_ylabel(y_label, fontsize=label_fontsize)
        #fig_ax[1].set_xlim(left=0, right=500)
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

    def draw_relation(self, x, y, weight, fig_ax=None, title=None,
                       x_label="# of followers", y_label="# of followings",
                       style='b-', marker='.'):
        if not fig_ax:
            fig_ax = plt.subplots(figsize=figsize)# print("Degree sequence", degree_sequence)
        x, y, weight = np.array(x), np.array(y), np.array(weight)
        from math import log, ceil, exp
        x_bins = [exp(x) for x in range(ceil(log(x.max()))+1)]
        y_bins = [exp(x) for x in range(ceil(log(y.max()))+1)]
        H, xedges, yedges = np.histogram2d(x, y, [x_bins, y_bins], weights=weight)
        H = H.T
        X, Y = np.meshgrid(xedges, yedges)
        ax = fig_ax[1].pcolormesh(X, Y, H)
        #ax = fig_ax[1].scatter(x, y, s=100, c=weight, alpha=0.5)#, style, marker=marker, linewidth=linewidth)
        fig_ax[0].colorbar(ax)
        fig_ax[1].set_xlabel(x_label, fontsize=label_fontsize)
        fig_ax[1].set_ylabel(y_label, fontsize=label_fontsize)
        fig_ax[1].tick_params(size=tick_fontsize)
        #fig_ax[1].set_ylim(bottom=1, top=5000)
        #fig_ax[1].set_xlim(left=1, right=40000)
        fig_ax[1].set_yscale('log', basey=10)
        fig_ax[1].set_xscale('log', basex=10)

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

