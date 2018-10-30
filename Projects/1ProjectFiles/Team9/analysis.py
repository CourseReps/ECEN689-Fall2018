"""
Module for performing analysis/plotting for the report.
"""

########################################################################
# IMPORTS

# Standard library:
import json

# Installed packages:
import pandas as pd
import numpy as np
import plotly.offline
import plotly.figure_factory as ff
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

# Project imports:
import read_atlas_data
import read_irs

########################################################################
# TWEAK MATPLOTLIB FOR IEEE STYLE
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['figure.figsize'] = (3.5, 1.64)
# mpl.rcParams['lines.markersize'] = 2
# mpl.rcParams['lines.markeredgewidth'] = 0.5
mpl.rcParams['figure.dpi'] = 300

# Font manager for text boxes
FONT_PROPERTIES = {'family': 'serif', 'size': 5}

########################################################################
# FUNCTIONS


def main():
    """Main function"""
    # Read the IRS data
    irs_data = read_irs.get_irs_data()

    # Notify.
    print('IRS data loaded. Column descriptions:')
    print(json.dumps(read_irs.COLUMNS, indent=2))

    # Read Food Environment Atlas data
    food_county, food_state = read_atlas_data.read_data()
    print('Food Environment Atlas data loaded.')

    # Check to see if the county data has duplicate FIPS codes. Turns
    # out it doesn't.
    # county_duplicate_fips = food_county.duplicated()
    # print(county_duplicate_fips.any())

    # Join the IRS data and county Food Environment Atlas data by FIPS
    # code. Since the IRS data has multiple entries per FIPS code, we'll
    # join on the IRS data
    joined_data = irs_data.join(food_county.set_index('FIPS'), on='FIPS')

    # How many NaN's do we have?
    total_rows = joined_data.shape[0]
    nan_rows = joined_data.isnull().sum().max()
    joined_data.dropna(inplace=True)
    print('In the joined data, {} rows were be dropped out of {}.'.format(
        nan_rows, total_rows))

    # Create maps for the various health factors.
    # map_plots(joined_data)

    # Create scatter plots:

    # Plot pct obese vs. pct of tax returns filed in each agi_stub
    scatter_plots(joined_data, health_column='PCT_OBESE_ADULTS13',
                  income_column='N1_pct_of_FIPS', ylabel='Pct. Obese',
                  filename='obese_scatter_N1')

    # pct obese vs. pct of total people in each agi_stub
    scatter_plots(joined_data, health_column='PCT_OBESE_ADULTS13',
                  income_column='total_people_pct_of_FIPS',
                  ylabel='Pct. Obese',
                  filename='obese_scatter_total_people')

    # pct diabetes vs. pct of tax returns filed in each agi_stub
    scatter_plots(joined_data, health_column='PCT_DIABETES_ADULTS13',
                  income_column='N1_pct_of_FIPS', ylabel='Pct. Diabetes',
                  filename='diabetes_scatter_N1')

    # pct diabetes vs. pct of total people in each agi_stb
    scatter_plots(joined_data, health_column='PCT_DIABETES_ADULTS13',
                  income_column='total_people_pct_of_FIPS',
                  ylabel='Pct. Diabetes',
                  filename='diabetes_scatter_total_people')

    # Compute approximate mean and median income per person in each FIPS
    # code, and create scatter plots with diabetes/obesity.
    scatter_mean_medians(joined_data)

    pass


def map_plots(data):
    """Do map plotting.

    plotly reference: https://plot.ly/python/county-choropleth/
    """
    # Plot percentage of lowest income people.
    agi1_bool = data['agi_stub'] == 1
    agi1_data = data['total_people_pct_of_FIPS'][agi1_bool] * 100
    agi1_pct = (agi1_data).tolist()
    agi1_fips = data['FIPS'][agi1_bool].tolist()
    # agi_lin = (agi1_data.max() - agi1_data.min()) / 5
    # pct_bins = np.linspace(agi1_data.min() + agi_lin,
    #                        agi1_data.max() - agi_lin,
    #                        num=5).tolist()
    pct_bins = np.percentile(agi1_data, [20, 40, 60, 80]).tolist()

    # colors from http://colorbrewer2.org
    # colorscale = ['#f0f9e8', '#ccebc5', '#a8ddb5', '#7bccc4', '#4eb3d3',
    #               '#2b8cbe', '#08589e']
    # colorscale = ['#f0f9e8', '#ccebc5', '#a8ddb5', '#7bccc4', '#43a2ca',
    #               '#0868ac']
    colorscale = ['#f0f9e8', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac']
    fig = ff.create_choropleth(fips=agi1_fips, values=agi1_pct,
                               round_legend_values=True,
                               binning_endpoints=pct_bins,
                               colorscale=colorscale,
                               legend_title='Pct. of People in $1-$25k AGI '
                                            'Bracket')
    plotly.offline.plot(fig, filename='pct_agi1.html')

    # Plot diabetes. We can still use agi1_bool - the diabetes rate is
    # the same for all agi_stubs.
    diabetes_data = data['PCT_DIABETES_ADULTS13'][agi1_bool]
    diabetes_pct = diabetes_data.tolist()
    # We'll re-use the agi1_fips.
    # New pct_bins (max for diabetes is 23.5:
    # diabetes_lin = (diabetes_data.max() - diabetes_data.min()) / 5
    # pct_bins = np.linspace(diabetes_data.min() + diabetes_lin,
    #                        diabetes_data.max() - diabetes_lin, num=5).tolist()
    pct_bins = np.percentile(diabetes_data, [20, 40, 60, 80]).tolist()
    fig = ff.create_choropleth(fips=agi1_fips, values=diabetes_pct,
                               round_legend_values=True,
                               binning_endpoints=pct_bins,
                               colorscale=colorscale,
                               legend_title='Pct. of Adults with Diabetes')
    plotly.offline.plot(fig, filename='pct_diabetes.html')

    # Plot obesity.
    obesity_data = data['PCT_OBESE_ADULTS13'][agi1_bool]
    obesity_pct = obesity_data.tolist()
    # We'll re-use the agi1_fips.
    # New pct_bins (max for obesity is 47.6:
    # obesity_lin = (obesity_data.max() - obesity_data.min()) / 5
    # pct_bins = np.linspace(obesity_data.min() + obesity_lin,
    #                        obesity_data.max() - obesity_lin, num=5).tolist()
    pct_bins = np.percentile(obesity_data, [20, 40, 60, 80]).tolist()
    fig = ff.create_choropleth(fips=agi1_fips, values=obesity_pct,
                               round_legend_values=True,
                               binning_endpoints=pct_bins,
                               colorscale=colorscale,
                               legend_title='Pct. of Obese Adults')
    plotly.offline.plot(fig, filename='pct_obese.html')


def scatter_plots(data, health_column, income_column, ylabel, filename):
    """Method for creating scatter plots for each agi stub vs health"""

    # Initialize figure. Let's make it the whole width of the paper,
    # minus the 1" margins.
    fig = plt.figure(figsize=[7.5, 1.64])

    # Loop over all the agi stubs
    for s in range(1, 7):
        # Initialize axis.
        ax = plt.subplot(1, 6, s)

        # Get data for this agi_stub.
        agi_bool = data['agi_stub'] == s
        pct_in_fips = data[income_column][agi_bool] * 100

        # Get health data. NOTE: This could be factored out of the loop.
        health_data = data[health_column][agi_bool]

        ax.plot(pct_in_fips, health_data, linestyle='None', marker='.',
                markersize=1)

        # Compute correlation coefficients. NOTE: Spearman may be better
        # here, as our data isn't Gaussian.
        pr, _ = pearsonr(pct_in_fips, health_data)
        sr, _ = spearmanr(pct_in_fips, health_data)
        print('')
        corr_text = 'Pearson: {:.2f}\nSpearman: {:.2f}'.format(pr, sr)
        txt = AnchoredText(corr_text, loc='upper center', prop=FONT_PROPERTIES,
                           frameon=False, pad=0, borderpad=0.2)
        ax.add_artist(txt)

        ax.set_ylabel(ylabel)
        ax.set_xlabel('Pct. in AGI Bracket')
        t = 'AGI: {}'
        ax.set_title(t.format(read_irs.AGI_STUBS[s]))

        # # Perform polynomial fits for degrees 1 and 2, plot the best.
        # c1 = np.polyfit(pct_in_fips, health_data, deg=1)
        # p1 = np.poly1d(c1)
        # f1 = p1(pct_in_fips)
        # r2_1 = r2_score(health_data, f1)
        #
        # # Degree 2.
        # c2 = np.polyfit(pct_in_fips, health_data, deg=2)
        # p2 = np.poly1d(c2)
        # f2 = p2(pct_in_fips)
        # r2_2 = r2_score(health_data, f1)
        #
        # # Use the best r2 score for plotting.
        # if r2_1 > r2_2:
        #     p = p1
        # else:
        #     p = p2
        #
        # # Sweep over the x data to get y data for the best fit.
        # x = np.arange(0, max(pct_in_fips), 1)
        # y = p(x)
        # ax.plot(x, y, marker='None')

    # TODO: More layout tweaks:
    # - Could do a shared y-axis for all the figures.
    # - Always room for tweaking tight_layout parameters
    # - Consider figure title, with reduced subplot titles
    plt.tight_layout(pad=0.05, h_pad=0, w_pad=0.2)
    # TODO: Save .eps for report.
    plt.savefig(filename + '.png')
    plt.savefig(filename + '.eps', type='eps')

def scatter_mean_medians(data):
    """"""
    # Overall mean income per FIPS:
    totals = data[['FIPS', 'total_people', 'A00100']].groupby('FIPS').sum()
    totals['mean_agi_per_person'] = (totals['A00100'] * 1000
                                     / totals['total_people'])

    # Join data.
    data = data.join(totals['mean_agi_per_person'], on='FIPS')

    # Grab boolean for a single agi_stub to get health data.
    health_bool = data['agi_stub'] == 1

    # Scatter for diabetes
    fig, ax = plt.subplots(1, 1)
    ax.plot(data[health_bool]['mean_agi_per_person'],
            data[health_bool]['PCT_DIABETES_ADULTS13'],
            linestyle='None', marker='.', markersize=1)

    pr, _ = pearsonr(data[health_bool]['mean_agi_per_person'],
                     data[health_bool]['PCT_DIABETES_ADULTS13'])
    sr, _ = spearmanr(data[health_bool]['mean_agi_per_person'],
                      data[health_bool]['PCT_DIABETES_ADULTS13'])
    corr_text = 'Pearson: {:.2f}\nSpearman: {:.2f}'.format(pr, sr)
    txt = AnchoredText(corr_text, loc='upper center', prop=FONT_PROPERTIES,
                       frameon=False, pad=0, borderpad=0.2)
    ax.add_artist(txt)

    ax.set_ylabel('Pct. Diabetes')
    ax.set_xlabel('Mean AGI per Person ($)')
    plt.tight_layout(pad=0.05, h_pad=0, w_pad=0)
    plt.savefig('mean_agi_diabetes.png')
    plt.savefig('mean_agi_diabetes.eps')

    # Scatter for obesity
    fig, ax = plt.subplots(1, 1)
    ax.plot(data[health_bool]['mean_agi_per_person'],
            data[health_bool]['PCT_OBESE_ADULTS13'],
            linestyle='None', marker='.', markersize=1)

    pr, _ = pearsonr(data[health_bool]['mean_agi_per_person'],
                     data[health_bool]['PCT_OBESE_ADULTS13'])
    sr, _ = spearmanr(data[health_bool]['mean_agi_per_person'],
                      data[health_bool]['PCT_OBESE_ADULTS13'])
    corr_text = 'Pearson: {:.2f}\nSpearman: {:.2f}'.format(pr, sr)
    txt = AnchoredText(corr_text, loc='upper center', prop=FONT_PROPERTIES,
                       frameon=False, pad=0, borderpad=0.2)
    ax.add_artist(txt)

    ax.set_ylabel('Pct. Obese')
    ax.set_xlabel('Mean AGI per Person ($)')
    plt.tight_layout(pad=0.05, h_pad=0, w_pad=0)
    plt.savefig('mean_agi_obese.png')
    plt.savefig('mean_agi_obese.eps', type='eps')

    # Compute median of means.
    total_groups = data[['FIPS', 'agi_stub', 'total_people',
                         'agi_per_person']].groupby('FIPS')

    # Initialize DataFrame for holding the median mean data.
    med_mean = pd.DataFrame(0.0, columns=['median_mean_agi'],
                            index=data['FIPS'].unique())

    # Loop over the groups
    for name, group in total_groups:
        # Ensure group is sorted by agi_stub
        group.sort_values(by='agi_stub')
        # Get the median index.
        c = np.cumsum(group['total_people'].values)
        med_ind = np.searchsorted(c, c[-1]/2)
        median_mean_agi = group.iloc[med_ind]['agi_per_person'],
        med_mean.loc[group.iloc[0]['FIPS']] = median_mean_agi

    # Join.
    data = data.join(med_mean, on='FIPS')

    # Scatter plots.
    # Diabetes:
    fig, ax = plt.subplots(1, 1)
    ax.plot(data[health_bool]['median_mean_agi'],
            data[health_bool]['PCT_DIABETES_ADULTS13'],
            linestyle='None', marker='.', markersize=1)

    pr, _ = pearsonr(data[health_bool]['median_mean_agi'],
                     data[health_bool]['PCT_DIABETES_ADULTS13'])
    sr, _ = spearmanr(data[health_bool]['median_mean_agi'],
                      data[health_bool]['PCT_DIABETES_ADULTS13'])
    corr_text = 'Pearson: {:.2f}\nSpearman: {:.2f}'.format(pr, sr)
    txt = AnchoredText(corr_text, loc='upper center', prop=FONT_PROPERTIES,
                       frameon=False, pad=0, borderpad=0.2)
    ax.add_artist(txt)

    ax.set_ylabel('Pct. Diabetes')
    ax.set_xlabel('Median Mean AGI per Person ($)')
    plt.tight_layout(pad=0.05, h_pad=0, w_pad=0)
    plt.savefig('median_mean_agi_diabetes.png')
    plt.savefig('median_mean_agi_diabetes.eps', type='eps')

    # Obesity:
    fig, ax = plt.subplots(1, 1)
    ax.plot(data[health_bool]['median_mean_agi'],
            data[health_bool]['PCT_OBESE_ADULTS13'],
            linestyle='None', marker='.', markersize=1)

    pr, _ = pearsonr(data[health_bool]['median_mean_agi'],
                     data[health_bool]['PCT_OBESE_ADULTS13'])
    sr, _ = spearmanr(data[health_bool]['median_mean_agi'],
                      data[health_bool]['PCT_OBESE_ADULTS13'])
    corr_text = 'Pearson: {:.2f}\nSpearman: {:.2f}'.format(pr, sr)
    txt = AnchoredText(corr_text, loc='upper center', prop=FONT_PROPERTIES,
                       frameon=False, pad=0, borderpad=0.2)
    ax.add_artist(txt)

    ax.set_ylabel('Pct. Obese')
    ax.set_xlabel('Median Mean AGI per Person ($)')
    plt.tight_layout(pad=0.05, h_pad=0, w_pad=0)
    plt.savefig('median_mean_agi_obese.png')
    plt.savefig('median_mean_agi_obese.eps', type='eps')


########################################################################
# MAIN


if __name__ == '__main__':
    main()
