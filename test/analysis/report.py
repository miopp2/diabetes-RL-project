"""
Changes that have been made:
- added "ensemble_BG_24h": ensemble of BG of all patients for plot (24h)
- changes "ensemble_BG": adjustments to x-ticks
- added "ensemble_insulin_24h": ensemble of insulin of all patients for plot (24h)
- added "ensemble_insulin": ensemble of insulin of all patients for plot (24h)
- added "ensemble_CHO_24h": ensemble of CHO of all patients for plot (24h)
- changes "ensemblePlot": CGM removed and insulin added as subplot
- changes "percent_stats": time in range changed to BG ranges (xx<BG<xx)
"""

import glob
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
from matplotlib.collections import PatchCollection
# from pandas.plotting import lag_plot
import logging
plt.rcParams.update({'figure.max_open_warning': 0})

logger = logging.getLogger(__name__)


def ensemble_BG_24h(BG, sim_days, ax=None, plot_var=False, nstd=3):
    dailyBG = BG.iloc[0:480, :]
    dailyBG.reset_index(drop=True, inplace=True)
    for day in range(1, sim_days):
        start_idx = day * 480
        end_idx = start_idx + 480
        next_day = BG.iloc[start_idx:end_idx, :]
        next_day.reset_index(drop=True, inplace=True)
        dailyBG = pd.concat([dailyBG, next_day], axis=1)

    mean_curve = dailyBG.transpose().mean()
    std_curve = dailyBG.transpose().std()
    up_env = mean_curve + nstd * std_curve
    down_env = mean_curve - nstd * std_curve

    # t = BG.index.to_pydatetime()
    t = pd.to_datetime(BG.index[:480])
    if ax is None:
        fig, ax = plt.subplots(1)
    if plot_var and not std_curve.isnull().all():
        ax.fill_between(
            t, up_env, down_env, alpha=0.5, label='+/- {0}*std'.format(nstd))
    for p in dailyBG:
        ax.plot_date(
            t, dailyBG[p], '-', color='grey', alpha=0.2, lw=0.1, label='_nolegend_')
    ax.plot(t, mean_curve, lw=2, label='Mean Curve')
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))

    ax.axhline(70, c='green', linestyle='--', label='Hypoglycemia', lw=1)
    ax.axhline(180, c='red', linestyle='--', label='Hyperglycemia', lw=1)

    ax.set_xlim([t[0], t[-1]])
    ax.set_ylim([dailyBG.min().min() - 10, dailyBG.max().max() + 10])
    # ax.legend()
    ax.set_ylabel('Blood Glucose (mg/dl)')
    #     fig.autofmt_xdate()
    return ax


def ensemble_BG(BG, ax=None, plot_var=False, nstd=3):
    mean_curve = BG.transpose().mean()
    std_curve = BG.transpose().std()
    up_env = mean_curve + nstd * std_curve
    down_env = mean_curve - nstd * std_curve

    # t = BG.index.to_pydatetime()
    t = pd.to_datetime(BG.index)

    # calculation of simulation days
    td = t[-1] - t[0]
    sim_days = td.days

    if ax is None:
        fig, ax = plt.subplots(1)
    if plot_var and not std_curve.isnull().all():
        ax.fill_between(
            t, up_env, down_env, alpha=0.5, label='+/- {0}*std'.format(nstd))
    for p in BG:
        ax.plot_date(
            t, BG[p], '-', color='grey', alpha=0.2, lw=0.5, label='_nolegend_')
    ax.plot(t, mean_curve, lw=2, label='Mean Curve')
    # ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))

    ax.axhline(70, c='green', linestyle='--', label='Hypoglycemia', lw=1)
    ax.axhline(180, c='red', linestyle='--', label='Hyperglycemia', lw=1)

    # adding vertical line after each day
    start_time = t[0]
    hours_day = 24
    for day in range(sim_days):
        ax.axvline([start_time + timedelta(hours=day * hours_day)], c='grey', linestyle='--', lw=1)

    ax.set_xlim([t[0], t[-1]])
    ax.set_ylim([BG.min().min() - 10, BG.max().max() + 10])
    # ax.legend()
    ax.set_ylabel('Blood Glucose (mg/dl)')
    #     fig.autofmt_xdate()
    return ax


def ensemble_insulin_24h(Insulin, sim_days, ax=None, plot_var=False, nstd=3):
    dailyInsulin = Insulin.iloc[0:480, :]
    dailyInsulin.reset_index(drop=True, inplace=True)
    for day in range(1, sim_days):
        start_idx = day * 480
        end_idx = start_idx + 480
        next_day = Insulin.iloc[start_idx:end_idx, :]
        next_day.reset_index(drop=True, inplace=True)
        dailyInsulin = pd.concat([dailyInsulin, next_day], axis=1)

    mean_curve = dailyInsulin.transpose().mean()
    std_curve = dailyInsulin.transpose().std()
    up_env = mean_curve + nstd * std_curve
    down_env = mean_curve - nstd * std_curve

    # t = Insulin.index.to_pydatetime()
    t = pd.to_datetime(Insulin.index[:480])
    if ax is None:
        fig, ax = plt.subplots(1)
    if plot_var and not std_curve.isnull().all():
        ax.fill_between(
            t, up_env, down_env, alpha=0.5, label='+/- {0}*std'.format(nstd))
    for p in dailyInsulin:
        ax.plot_date(
            t, dailyInsulin[p], '-', color='grey', alpha=0.2, lw=0.1, label='_nolegend_')
    ax.plot(t, mean_curve, lw=2, label='Mean Curve')
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))

    ax.set_xlim([t[0], t[-1]])
    ax.set_ylim([dailyInsulin.min().min() - 0.1, dailyInsulin.max().max() + 0.1])
    # ax.legend()
    ax.set_ylabel('Insulin (U/min)')
    #     fig.autofmt_xdate()
    return ax


def ensemble_insulin(insulin, ax=None, plot_var=False, nstd=3):
    mean_curve = insulin.transpose().mean()
    std_curve = insulin.transpose().std()
    up_env = mean_curve + nstd * std_curve
    down_env = mean_curve - nstd * std_curve
    down_env[down_env < 0] = 0

    # t = insulin.index.to_pydatetime()
    t = pd.to_datetime(insulin.index)

    # calculation of simulation days
    td = t[-1] - t[0]
    sim_days = td.days

    if ax is None:
        fig, ax = plt.subplots(1)
    if plot_var and not std_curve.isnull().all():
        ax.fill_between(
            t, up_env, down_env, alpha=0.5, label='+/- {0}*std'.format(nstd))
    for p in insulin:
        ax.plot_date(
            t, insulin[p], '-', color='grey', alpha=0.2, lw=0.5, label='_nolegend_')
    ax.plot(t, mean_curve, lw=2, label='Mean Curve')
    # ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))

    # adding vertical line after each day
    start_time = t[0]
    hours_day = 24
    for day in range(sim_days):
        ax.axvline([start_time + timedelta(hours=day * hours_day)], c='grey', linestyle='--', lw=1)

    ax.set_xlim([t[0], t[-1]])
    ax.set_ylim([insulin.min().min() - 0.1, insulin.max().max() + 0.1])
    # ax.legend()
    ax.set_ylabel('Insulin (U/min)')
    #     fig.autofmt_xdate()
    return ax

def ensemble_CHO_24h(CHO, sim_days, ax=None, plot_var=False, nstd=3):
    dailyCHO = CHO.iloc[0:480, :]
    dailyCHO.reset_index(drop=True, inplace=True)
    for day in range(1, sim_days):
        start_idx = day * 480
        end_idx = start_idx + 480
        next_day = CHO.iloc[start_idx:end_idx, :]
        next_day.reset_index(drop=True, inplace=True)
        dailyCHO = pd.concat([dailyCHO, next_day], axis=1)

    mean_curve = dailyCHO.transpose().mean()
    std_curve = dailyCHO.transpose().std()
    up_env = mean_curve + nstd * std_curve
    down_env = mean_curve - nstd * std_curve

    # t = CHO.index.to_pydatetime()
    t = pd.to_datetime(CHO.index[:480])
    if ax is None:
        fig, ax = plt.subplots(1)
    if plot_var and not std_curve.isnull().all():
        ax.fill_between(
            t, up_env, down_env, alpha=0.5, label='+/- {0}*std'.format(nstd))
    for p in dailyCHO:
        ax.plot_date(
            t, dailyCHO[p], '-', color='grey', alpha=0.2, lw=0.1, label='_nolegend_')
    ax.plot(t, mean_curve, lw=2, label='Mean Curve')
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))

    ax.set_xlim([t[0], t[-1]])
    ax.set_ylim([dailyCHO.min().min() - 0.1, dailyCHO.max().max() + 0.1])
    # ax.legend()
    ax.set_ylabel('CHO (U/min)')
    #     fig.autofmt_xdate()
    return ax

def ensemblePlot(df):
    df_BG = df.unstack(level=0).BG
    df_insulin = df.unstack(level=0).insulin
    df_CHO = df.unstack(level=0).CHO
    fig = plt.figure(figsize=(8, 6), dpi=600, layout='constrained')  # figure size and image quality
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1 = ensemble_BG(df_BG, ax=ax1, plot_var=True, nstd=1)
    ax2 = ensemble_insulin(df_insulin, ax=ax2, plot_var=True, nstd=1)
    # t = df_CHO.index.to_pydatetime()
    t = pd.to_datetime(df_CHO.index)
    ax3.plot(t, df_CHO, '#1f77b4')

    ax1.tick_params(labelbottom=False)  # add/remove current day as tic
    ax2.tick_params(labelbottom=False)  # add/remove current day as tic
    ax3.xaxis.set_minor_locator(mdates.AutoDateLocator())
    ax3.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax3.xaxis.set_major_locator(mdates.DayLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
    ax3.set_xlim([t[0], t[-1]])
    ax1.set_ylabel('Blood Glucose (mg/dl)')
    ax2.set_ylabel('Insulin (U/min)')
    ax3.set_ylabel('CHO (g)')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    return fig, ax1, ax2, ax3


def ensemblePlot_24h(df, sim_days):
    df_BG = df.unstack(level=0).BG
    df_insulin = df.unstack(level=0).insulin
    df_CHO = df.unstack(level=0).CHO
    fig = plt.figure(figsize=(8, 6), dpi=600, layout='constrained')  # figure size and image quality
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1 = ensemble_BG_24h(df_BG, sim_days=sim_days, ax=ax1, plot_var=True, nstd=1)
    ax2 = ensemble_insulin_24h(df_insulin, sim_days=sim_days, ax=ax2, plot_var=True, nstd=1)
    ax3 = ensemble_CHO_24h(df_CHO, sim_days=sim_days, ax=ax3, plot_var=True, nstd=1)
    # t = df_CHO.index.to_pydatetime()
    t = pd.to_datetime(df_CHO.index)

    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)
    ax3.xaxis.set_minor_locator(mdates.AutoDateLocator())
    ax3.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax3.xaxis.set_major_locator(mdates.DayLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter(''))
    ax3.set_xlim([t[0], t[480]])
    ax1.set_ylabel('Blood Glucose (mg/dl)')
    ax2.set_ylabel('Insulin (U/min)')
    ax3.set_ylabel('CHO (g)')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3,1))
    return fig, ax1, ax2, ax3


def percent_stats(BG, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    # exclude BG > 50
    p_hyper = ((BG > 180) & (BG < 250)).sum() / len(BG) * 100
    p_hyper.name = '250<BG>180'
    # exclude BG < 50
    p_hypo = ((BG < 70) & (BG > 50)).sum() / len(BG) * 100
    p_hypo.name = '70<BG>50'
    p_normal = ((BG >= 70) & (BG <= 180)).sum() / len(BG) * 100
    p_normal.name = '70<=BG<=180'
    p_250 = (BG > 250).sum() / len(BG) * 100
    p_250.name = 'BG>250'
    p_50 = (BG < 50).sum() / len(BG) * 100
    p_50.name = 'BG<50'
    p_stats = pd.concat([p_normal, p_hyper, p_hypo, p_250, p_50], axis=1)
    # p_stats.plot(ax=ax, kind='bar')
    p_stats.plot.bar(ax=ax, stacked=True)
    ax.set_ylabel('Percent of time in Range (%)')
    fig.tight_layout()
    #     p_stats.transpose().plot(kind='bar', legend=False)
    return p_stats, fig, ax


def risk_index_trace(df_BG, visualize=False):
    chunk_BG = [df_BG.iloc[i:i + 60, :] for i in range(0, len(df_BG), 60)]

    fBG = [
        np.mean(1.509 * (np.log(BG[BG > 0]) ** 1.084 - 5.381)) for BG in chunk_BG
    ]

    fBG_df = pd.concat(fBG, axis=1).transpose()

    LBGI = 10 * (fBG_df * (fBG_df < 0)) ** 2
    HBGI = 10 * (fBG_df * (fBG_df > 0)) ** 2
    RI = LBGI + HBGI

    ri_per_hour = pd.concat(
        [LBGI.transpose(), HBGI.transpose(),
         RI.transpose()],
        keys=['LBGI', 'HBGI', 'Risk Index'])

    axes = []
    if visualize:
        logger.info('Plotting risk trace plot')
        ri_per_hour_plot = pd.concat(
            [HBGI.transpose(), -LBGI.transpose()], keys=['HBGI', '-LBGI'])
        for i in range(len(ri_per_hour_plot.unstack(level=0))):
            logger.debug(
                ri_per_hour_plot.unstack(level=0).iloc[i].unstack(level=1))
            axtmp = ri_per_hour_plot.unstack(level=0).iloc[i].unstack(
                level=1).plot.bar(stacked=True)
            axes.append(axtmp)
            plt.xlabel('Time (hour)')
            plt.ylabel('Risk Index')

    ri_mean = ri_per_hour.transpose().mean().unstack(level=0)
    fig, ax = plt.subplots(1)
    ri_mean.plot(ax=ax, kind='bar')
    fig.tight_layout()

    axes.append(ax)
    return ri_per_hour, ri_mean, fig, axes


def CVGA_background(ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)

    ax.set_xlim(109, 49)
    ax.set_ylim(105, 405)
    ax.set_xticks([110, 90, 70, 50])
    ax.set_yticks([110, 180, 300, 400])
    ax.set_xticklabels(['110', '90', '70', '<50'])
    ax.set_yticklabels(['110', '180', '300', '>400'])
    #     fig.suptitle('Control Variability Grid Analysis (CVGA)')
    ax.set_title('Control Variability Grid Analysis (CVGA)')
    ax.set_xlabel('Min BG (2.5th percentile)')
    ax.set_ylabel('Max BG (97.5th percentile)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    rectangles = {
        'A-Zone': plt.Rectangle((90, 110), 20, 70, color='limegreen'),
        'Lower B': plt.Rectangle((70, 110), 20, 70, color='green'),
        'Upper B': plt.Rectangle((90, 180), 20, 120, color='green'),
        'B-Zone': plt.Rectangle((70, 180), 20, 120, color='green'),
        'Lower C': plt.Rectangle((50, 110), 20, 70, color='yellow'),
        'Upper C': plt.Rectangle((90, 300), 20, 100, color='yellow'),
        'Lower D': plt.Rectangle((50, 180), 20, 120, color='orange'),
        'Upper D': plt.Rectangle((70, 300), 20, 100, color='orange'),
        'E-Zone': plt.Rectangle((50, 300), 20, 100, color='red')
    }
    facecolors = [rectangles[r].get_facecolor() for r in rectangles]
    pc = PatchCollection(
        rectangles.values(),
        facecolor=facecolors,
        edgecolors='w',
        lw=2,
        alpha=1)
    ax.add_collection(pc)
    for r in rectangles:
        rx, ry = rectangles[r].get_xy()
        cx = rx + rectangles[r].get_width() / 2.0
        cy = ry + rectangles[r].get_height() / 2.0
        if r in ['Lower B', 'Upper B', 'B-Zone']:
            ax.annotate(
                r, (cx, cy),
                weight='bold',
                color='w',
                fontsize=10,
                ha='center',
                va='center')
        else:
            ax.annotate(
                r, (cx, cy),
                weight='bold',
                color='k',
                fontsize=10,
                ha='center',
                va='center')

    return fig, ax


def CVGA_analysis(BG):
    BG_min = np.percentile(BG, 2.5, axis=0)
    BG_max = np.percentile(BG, 97.5, axis=0)
    BG_min[BG_min < 50] = 50
    BG_min[BG_min > 400] = 400
    BG_max[BG_max < 50] = 50
    BG_max[BG_max > 400] = 400

    perA = ((BG_min > 90) & (BG_min <= 110) & (BG_max >= 110)
            & (BG_max < 180)).sum() / float(len(BG_min))
    perB = ((BG_min > 70) & (BG_min <= 110) & (BG_max >= 110)
            & (BG_max < 300)).sum() / float(len(BG_min)) - perA
    perC = (((BG_min > 90) & (BG_min <= 110) & (BG_max >= 300)) |
            ((BG_min <= 70) & (BG_max >= 110) &
             (BG_max < 180))).sum() / float(len(BG_min))
    perD = (((BG_min > 70) & (BG_min <= 90) & (BG_max >= 300)) |
            ((BG_min <= 70) & (BG_max >= 180) &
             (BG_max < 300))).sum() / float(len(BG_min))
    perE = ((BG_min <= 70) & (BG_max >= 300)).sum() / float(len(BG_min))
    return BG_min, BG_max, perA, perB, perC, perD, perE


def CVGA(BG_list, label=None):
    if not isinstance(BG_list, list):
        BG_list = [BG_list]
    if not isinstance(label, list):
        label = [label]
    if label is None:
        label = ['BG%d' % (i + 1) for i in range(len(BG_list))]
    fig, ax = CVGA_background()
    zone_stats = []
    for (BG, l) in zip(BG_list, label):
        BGmin, BGmax, A, B, C, D, E = CVGA_analysis(BG)
        ax.scatter(
            BGmin,
            BGmax,
            edgecolors='k',
            zorder=4,
            label='%s (A: %d%%, B: %d%%, C: %d%%, D: %d%%, E: %d%%)' %
                  (l, 100 * A, 100 * B, 100 * C, 100 * D, 100 * E))
        zone_stats.append((A, B, C, D, E))

    zone_stats = pd.DataFrame(zone_stats, columns=['A', 'B', 'C', 'D', 'E'])
    #     ax.legend(bbox_to_anchor=(1, 1.10), borderaxespad=0.5)
    ax.legend()
    return zone_stats, fig, ax


def report(df, save_path=None, sim_days=1):
    BG = df.unstack(level=0).BG

    fig_ensemble, ax1, ax2, ax3 = ensemblePlot(df)
    fig_ensemble24, ax6, ax7, ax8 = ensemblePlot_24h(df, sim_days=sim_days)
    pstats, fig_percent, ax4 = percent_stats(BG)
    ri_per_hour, ri_mean, fig_ri, ax5 = risk_index_trace(BG, visualize=False)
    # zone_stats, fig_cvga, ax6 = CVGA(BG, label='')
    # axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
    # figs = [fig_ensemble, fig_percent, fig_ri, fig_cvga]
    figs = [fig_ensemble, fig_percent, fig_ri, fig_ensemble24]
    results = pd.concat([pstats, ri_mean], axis=1)

    if save_path is not None:
        results.to_csv(os.path.join(save_path, 'performance_stats.csv'))
        ri_per_hour.to_csv(os.path.join(save_path, 'risk_trace.csv'))
        # zone_stats.to_csv(os.path.join(save_path, 'CVGA_stats.csv'))

        fig_ensemble.savefig(os.path.join(save_path, 'BG_trace.png'))
        fig_ensemble24.savefig(os.path.join(save_path, 'BG_trace_24h.png'))
        fig_percent.savefig(os.path.join(save_path, 'zone_stats.png'))
        fig_ri.savefig(os.path.join(save_path, 'risk_stats.png'))
        # fig_cvga.savefig(os.path.join(save_path, 'CVGA.png'))

    # plt.show()
    # return results, ri_per_hour, zone_stats, figs, axes
    return results, ri_per_hour, figs, axes


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('analysis.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - \n %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    # logger.addHandler(fh)
    logger.addHandler(ch)
    # For test only
    path = os.path.join('..', 'results', '2022-12-17_10-13-30', 'BBController')
    os.chdir(path)
    filename = glob.glob('*#*.csv')
    name = [_f[:-4] for _f in filename]
    df = pd.concat([pd.read_csv(f, index_col=0) for f in filename], keys=name)
    # sim_days = 1
    sim_days = int(len(df) / len(filename) / 480) # overlays multiple days on one day
    if df['insulin'].dtypes == 'object':
        df['insulin'] = df['insulin'].str.replace(r'[\[\]]', '', regex=True).astype(float)
    results, ri_per_hour, zone_stats, axes = report(df, '..\\BBController', sim_days=sim_days)
    # print results
    # # print ri_per_hour
    # print zone_stats
