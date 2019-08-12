import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import os
import glob
import ludwig
import re
from scipy import stats
from time_series_analysis import TimeSeriesAnalysis
import csv

# Find all intervals of interval ms where change > num*stdev
def get_relevant_intervals(interval, data, num):
    stdev = np.std(data)
    rel = []
    for i in range(len(data) - interval):
        if (np.abs(data[i] - data[i+interval]) > num*stdev):
            rel.append(i)
    return rel

def plot_filtered_results(data, interval, num, title):
    filtered_data = scipy.signal.medfilt(data, interval+1)

    relevant_points = get_relevant_intervals(interval, data, num)


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title(title)
    ax.plot(range(len(data)), data, 'y')
    ax.plot(range(len(data)), filtered_data, 'g', linewidth=1.0,zorder=2)

    for i in relevant_points:
        ax.plot(range(i, i+interval), filtered_data[i:i+interval], 'm', linewidth=5.0, zorder=1)

    plt.show()


def save_relevant_intervals_with_action_types(lr, jl, ls, bl, lcd, br, dates, interval, num, filename, action_types, action_dates, utterances=[],utterance_dates=[]):
    
    f, axarr = plt.subplots(6, sharex=True)
    for i,data in enumerate([lr, jl, ls, bl, lcd, br]):
        filtered_data = scipy.signal.medfilt(data, interval+1)
        relevant_points = get_relevant_intervals(interval, data, num)
        data_line, = axarr[i].plot(dates, data, 'y', label='Data')
        filtered_data_line,  = axarr[i].plot(dates, filtered_data, 'g', linewidth=1.0, zorder=2, label='Filtered data')

        for pt in relevant_points:
            rel_pt_line,= axarr[i].plot(dates[pt:pt+interval], filtered_data[pt:pt+interval], 'm', linewidth=5.0, zorder=1, label='Intervals')

            axarr[i].set_ylim([-1,1])

        for action, date in zip(action_types, action_dates):
            axarr[i].axvline(x=date, color='b', linewidth=2, zorder=0, clip_on=False)
            # if action == 'WAVE':
            #     axarr[i].axvline(x=date, color='r', linewidth=2, zorder=0, clip_on=False, label='WAVE action', )
            # elif action == 'HAPPY':
            #     axarr[i].axvline(x=date, color='r', linewidth=2, zorder=0, clip_on=False, label='SMILE action')
            # elif action == 'LOOK_RIGHT':
            #     axarr[i].axvline(x=date, color='r', linewidth=2, zorder=0, clip_on=False, label='LOOK RIGHT action')
        for utterance_date in utterance_dates:
            axarr[i].axvline(x=utterance_date, color='r', linewidth=2, zorder=0, clip_on=False)
        axarr[i].set_ylim([-1,1])
    #f.legend(handles=[data_line, filtered_data_line, rel_pt_line], labels=['Data', 'Filtered data', 'Relevant intervals'], loc='right')

    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    # ax.set_title(title)
    # ax.plot(dates, data, 'y')
    # ax.plot(dates, filtered_data, 'g', linewidth=1.0, zorder=2)

    # for i in relevant_points:
    #     ax.plot(dates[i:i+interval], filtered_data[i:i+interval], 'm', linewidth=5.0, zorder=1)

    # for action,date in zip(action_types, action_dates):
    #     if action == 'WAVE':
    #         ax.axvline(x=date, color='b', linewidth=2, zorder=0, linestyle='dashed')
    #     elif action == 'HAPPY':
    #         ax.axvline(x=date, color='r', linewidth=2, zorder=0, linestyle='dashed')
    #     elif action == 'LOOK_RIGHT':
    #         ax.axvline(x=date, color='k', linewidth=2, zorder=0, linestyle='dashed')
    # ax.set_ylim([-1,1])
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()
    plt.savefig(filename)

def get_robot_actions(id):
    session_directory = '/p/spoclab/data/Ludwig/ROBOT_DATA/Session2/'
    utterances = []
    actions = []
    utterance_dates = []
    action_dates = []
    for order in [1, 2, 3]:
        robot_file = glob.glob(session_directory + str(id) + '/RobotActions*order-' + str(order) + '*.txt')
        [utterances_per_image, actions_per_image, utterance_dates_per_image, action_dates_per_image] = ludwig.extract_robot_actions(robot_file[0])
        utterances += utterances_per_image
        actions += actions_per_image
        utterance_dates += utterance_dates_per_image
        action_dates += actions_per_image
    return utterances, actions, utterance_dates, action_dates


def get_changepoints():
    session_directory = '/p/spoclab/data/Ludwig/ROBOT_DATA/Session2/'
    for id in [3, 9]:
        lip_raiser_data = []
        jaw_lower_data = []
        lip_stretcher_data = []
        brow_lower_data = []
        lip_corner_depressor_data = []
        brow_raiser_data = []
        au_dates = []
        utterances = []
        actions = []
        utterance_dates = []
        action_dates = []
        action_types = []
        for order in [1, 2, 3]:
            au_file = glob.glob(session_directory + str(id) + '/AU_session*order-' + str(order) + '*.txt')
            robot_file = glob.glob(session_directory + str(id) + '/RobotActions*order-' + str(order) + '*.txt')
            [lip_raiser_data_per_image, jaw_lower_data_per_image, lip_stretcher_data_per_image, brow_lower_data_per_image, lip_corner_depressor_data_per_image, brow_raiser_data_per_image, au_dates_per_image] = ludwig.extract_animation_units(au_file[0])
            [utterances_per_image, actions_per_image, utterance_dates_per_image, action_dates_per_image] = ludwig.extract_robot_actions(robot_file[0])
            action_types_per_image = ludwig.classify_robot_actions(actions_per_image)

            lip_raiser_data += lip_stretcher_data_per_image
            jaw_lower_data += jaw_lower_data_per_image
            lip_stretcher_data += lip_stretcher_data_per_image
            brow_lower_data += brow_lower_data_per_image
            lip_corner_depressor_data += lip_corner_depressor_data_per_image
            brow_raiser_data += brow_raiser_data_per_image
            au_dates += au_dates_per_image
            utterances += utterances_per_image
            actions += actions_per_image
            utterance_dates += utterance_dates_per_image
            action_dates += actions_per_image
            action_types += action_types_per_image

        changepoints_file = 'changepoints_P%d.csv' % id
        with open(changepoints_file, 'w') as cfile:
            csvwriter = csv.writer(cfile)

            for i, data in enumerate([lip_raiser_data, jaw_lower_data, lip_stretcher_data, brow_lower_data, lip_corner_depressor_data, brow_raiser_data]):
                filtered_data = scipy.signal.medfilt(data, 101)
                ts = TimeSeriesAnalysis(au_dates, filtered_data)
                ts.ChangePointAnalysis()
                changepoints = [el[0] for el in ts.changepoints]
                changepoints = [au_dates[cp] for cp in changepoints]
                csvwriter.writerow(changepoints)


def plot_animation_units():
    session_directory = '/p/spoclab/data/Ludwig/ROBOT_DATA/Session2/'
    for id in [3, 9]:
        lip_raiser_data = []
        jaw_lower_data = []
        lip_stretcher_data = []
        brow_lower_data = []
        lip_corner_depressor_data = []
        brow_raiser_data = []
        au_dates = []
        utterances = []
        actions = []
        utterance_dates = []
        action_dates = []
        action_types = []
        for order in [1, 2, 3]:
            au_file = glob.glob(session_directory + str(id) + '/AU_session*order-' + str(order) + '*.txt')
            robot_file = glob.glob(session_directory + str(id) + '/RobotActions*order-' + str(order) + '*.txt')
            [lip_raiser_data_per_image, jaw_lower_data_per_image, lip_stretcher_data_per_image, brow_lower_data_per_image, lip_corner_depressor_data_per_image, brow_raiser_data_per_image, au_dates_per_image] = ludwig.extract_animation_units(au_file[0])
            [utterances_per_image, actions_per_image, utterance_dates_per_image, action_dates_per_image] = ludwig.extract_robot_actions(robot_file[0])
            action_types_per_image = ludwig.classify_robot_actions(actions_per_image)

            lip_raiser_data += lip_stretcher_data_per_image
            jaw_lower_data += jaw_lower_data_per_image
            lip_stretcher_data += lip_stretcher_data_per_image
            brow_lower_data += brow_lower_data_per_image
            lip_corner_depressor_data += lip_corner_depressor_data_per_image
            brow_raiser_data += brow_raiser_data_per_image
            au_dates += au_dates_per_image
            utterances += utterances_per_image
            actions += actions_per_image
            utterance_dates += utterance_dates_per_image
            action_dates += actions_per_image
            action_types += action_types_per_image
        filename = ('animation_units_P%d.png' % id)

        # Normalize dates:
        norm = min(au_dates)
        au_dates = [aud - norm for aud in au_dates]
        utterance_dates = [ud - norm for ud in utterance_dates]
        action_dates = [ad - norm for ad in action_dates]

        f, axarr = plt.subplots(6, sharex=True)
        for i, data in enumerate([lip_raiser_data, jaw_lower_data, lip_stretcher_data, brow_lower_data, lip_corner_depressor_data, brow_raiser_data]):
            filtered_data = scipy.signal.medfilt(data, 101)
            ts = TimeSeriesAnalysis(au_dates, filtered_data)
            ts.ChangePointAnalysis()
            changepoints = [el[0] for el in ts.changepoints]

            data_line, = axarr[i].plot(au_dates, data, 'y', label='Data')
            filtered_data_line,  = axarr[i].plot(au_dates, filtered_data, 'g', linewidth=1.0, zorder=2, label='Filtered data')

            for changepoint in changepoints:
                axarr[i].plot(au_dates[changepoint], filtered_data[changepoint], 'bo')

            for action, action_date in zip(actions, action_dates):
                axarr[i].axvline(x=action_date, color='m', linewidth=2, zorder=0, clip_on=False)

            for utterance_date in utterance_dates:
                axarr[i].axvline(x=utterance_date, color='m', linewidth=2, zorder=0, clip_on=False)
            axarr[i].set_ylim([-1,1])
            axarr[i].set_xlim([au_dates[0],au_dates[-1]])
            axarr[i].axes.yaxis.set_ticklabels([])

        axarr[0].set_title('Lip raiser')
        axarr[1].set_title('Jaw lower')
        axarr[2].set_title('Lip stretcher')
        axarr[3].set_title('Brow lower')
        axarr[4].set_title('Lip corner depressor')
        axarr[5].set_title('Brow raiser')

        axarr[3].set_ylabel('Animation units')
        axarr[5].set_xlabel('Time (s)')
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

        f.subplots_adjust(hspace=.5)
        #plt.show(f)
        plt.savefig(filename)

        #save_relevant_intervals_with_action_types(lip_raiser_data, jaw_lower_data, lip_stretcher_data, brow_lower_data, lip_corner_depressor_data, brow_raiser_data, au_dates, 100, 3, filename, action_types, action_dates, utterances, utterance_dates)


def plot_all_animation_units_session2():
    session_directory = '/p/spoclab/data/Ludwig/ROBOT_DATA/Session2/'
    for id in os.listdir(session_directory):
        au_files = glob.glob(session_directory + str(id) + '/AU_session*')
        robot_files = glob.glob(session_directory + str(id) + '/RobotActions*')

        for au_file, robot_file in zip(au_files, robot_files):
            filename = re.search('(AU_.*)\.txt', au_file).groups()[0] + '_actions.png'
            [lip_raiser_data, jaw_lower_data, lip_stretcher_data, brow_lower_data, lip_corner_depressor_data, brow_raiser_data, au_dates] = ludwig.extract_animation_units(au_file)
            [utterances, actions, utterance_dates, action_dates] = ludwig.extract_robot_actions(robot_file)
            action_types = ludwig.classify_robot_actions(actions)
            save_relevant_intervals_with_action_types(lip_raiser_data, jaw_lower_data, lip_stretcher_data, brow_lower_data, lip_corner_depressor_data, brow_raiser_data, au_dates, 100, 3, filename, action_types, action_dates)

def plot_all_animation_units_session1():
    session_directory = '/p/spoclab/data/Ludwig/ROBOT_DATA/Session1/'
    for id in ['1', '11_1', '3', '5', '7', '9', '10_1', '2', '4', '6', '8']:
        au_files = glob.glob(session_directory + str(id) + '/AU_session*')

        for au_file in au_files:
            filename = re.search('(AU_.*)\.txt', au_file).groups()[0] + '_actions.png'
            [lip_raiser_data, jaw_lower_data, lip_stretcher_data, brow_lower_data, lip_corner_depressor_data, brow_raiser_data, au_dates] = ludwig.extract_animation_units(au_file)
            save_relevant_intervals_with_action_types(lip_raiser_data, jaw_lower_data, lip_stretcher_data, brow_lower_data, lip_corner_depressor_data, brow_raiser_data, au_dates, 100, 3, filename, [], [])

''' Return ordered session files '''
def get_session_files(session, uid):

    # P10 and P11 are labeled as P10_1 and P11_1 
    if (os.path.exists('/p/spoclab/data/Ludwig/ROBOT_DATA/Session%d/%d/' % (session, uid))):
        au_session_files_template = '/p/spoclab/data/Ludwig/ROBOT_DATA/Session%d/%d/AU_session*' % (session, uid)
        au_session_files_order_template = '/p/spoclab/data/Ludwig/ROBOT_DATA/Session%d/%d/' % (session, uid)
    else:
        au_session_files_template = '/p/spoclab/data/Ludwig/ROBOT_DATA/Session%d/%d_%d/AU_session*' % (session, uid, session)
        au_session_files_order_template = '/p/spoclab/data/Ludwig/ROBOT_DATA/Session%d/%d_%d/' % (session, uid, session)

    num_files = len(glob.glob(au_session_files_template))
    session_files = []

    # We only want to return 3 files
    for order in range(1, min(num_files + 1, 4)):
        session_files.append('%sAU_session-%d_id-%d_order-%d_0.txt' % (au_session_files_order_template, session, uid, order))
    return session_files

def get_au_per_speaker(session_files):
    session_lr = []
    session_jl = []
    session_ls = []
    session_bl = []
    session_lcd = []
    session_br = []
    session_dates = []

    for session_file in session_files:
        [lip_raiser_data, jaw_lower_data, lip_stretcher_data, brow_lower_data, lip_corner_depressor_data, brow_raiser_data, au_dates] = ludwig.extract_animation_units(session_file)
        session_lr += lip_raiser_data
        session_jl += jaw_lower_data
        session_ls += lip_stretcher_data
        session_bl += brow_lower_data
        session_lcd += lip_corner_depressor_data
        session_br += brow_raiser_data
        session_dates += au_dates
    return session_lr, session_jl, session_ls, session_bl, session_lcd, session_br, session_dates

def cross_correlate_animation_units():
    ids = ['2', '3', '4', '5', '9']
    ids = [2,3,4,5,9]
    session1_dir = '/p/spoclab/data/Ludwig/ROBOT_DATA/Session1/'
    session2_dir = '/p/spoclab/data/Ludwig/ROBOT_DATA/Session2/'

    for id in ids:
        session1_files = get_session_files(1, id)
        session2_files = get_session_files(2, id)

        session1_lr, session1_jl, session1_ls, session1_bl, session1_lcd, session1_br, session1_dates = get_au_per_speaker(session1_files)

        session2_lr, session2_jl, session2_ls, session2_bl, session2_lcd, session2_br, session2_dates = get_au_per_speaker(session2_files)


        print('ID: ' + str(id))
        print('Session1 | Session 2 | ttest  ')
        print('Lip raiser: mean=%f, std=%f | mean=%f, std=%f | %s' % (np.mean(session1_lr), np.std(session1_lr), np.mean(session2_lr), np.std(session2_lr), stats.ttest_ind(session1_lr, session2_lr)))
        #print(np.correlate(session1_lr, session2_lr))
        print('Jaw lower: mean=%f, std=%f | mean=%f, std=%f | %s' % (np.mean(session1_jl), np.std(session1_jl), np.mean(session2_jl), np.std(session2_jl), stats.ttest_ind(session1_jl, session2_jl)))
        #print(np.correlate(session1_jl, session2_jl))
        print('Lip stretcher: mean=%f, std=%f | mean=%f, std=%f | %s' % (np.mean(session1_ls), np.std(session1_ls), np.mean(session2_ls), np.std(session2_ls), stats.ttest_ind(session1_ls, session2_ls)))
        #print(np.correlate(session1_ls, session2_ls))
        print('Brow lower: mean=%f, std=%f | mean=%f, std=%f | %s' % (np.mean(session1_bl), np.std(session1_bl), np.mean(session2_bl), np.std(session2_bl), stats.ttest_ind(session1_bl, session2_bl)))
        #print(np.correlate(session1_bl, session2_bl))
        print('Lip corner depressor: mean=%f, std=%f | mean=%f, std=%f | %s' % (np.mean(session1_lcd), np.std(session1_lcd), np.mean(session2_lcd), np.std(session2_lcd), stats.ttest_ind(session1_lcd, session2_lcd)))
        #print(np.correlate(session1_lcd, session2_lcd))
        print('Brow raiser: mean=%f, std=%f | mean=%f, std=%f | %s' % (np.mean(session1_br), np.std(session1_br), np.mean(session1_br), np.std(session1_br), stats.ttest_ind(session1_br, session2_br)))
        #print(np.correlate(session1_br, session2_br))
        print('')

def get_means_and_std():
    ids = [3,9]
    for id in ids:
        session1_files = get_session_files(1, id)
        session2_files = get_session_files(2, id)

        session1_lr, session1_jl, session1_ls, session1_bl, session1_lcd, session1_br, session1_dates = get_au_per_speaker(session1_files)

        session2_lr, session2_jl, session2_ls, session2_bl, session2_lcd, session2_br, session2_dates = get_au_per_speaker(session2_files)

        print(str(id))
        print('& LR & %.4f (%.4f) & %.4f (%.4f) & %g \\\\ \cline{2-4}' % (np.mean(session1_lr), np.std(session1_lr), np.mean(session2_lr), np.std(session2_lr), stats.ttest_ind(session1_lr, session2_lr)[1]))
        print('& JL & %.4f (%.4f) & %.4f (%.4f) & %g \\\\ \cline{2-4}' % (np.mean(session1_jl), np.std(session1_jl), np.mean(session2_jl), np.std(session2_jl), stats.ttest_ind(session1_jl, session2_jl)[1]))
        print('& LS & %.4f (%.4f) & %.4f (%.4f) & %g \\\\ \cline{2-4}' % (np.mean(session1_ls), np.std(session1_ls), np.mean(session2_ls), np.std(session2_ls), stats.ttest_ind(session1_ls, session2_ls)[1]))
        print('& BL & %.4f (%.4f) & %.4f (%.4f) & %g \\\\ \cline{2-4}' % (np.mean(session1_bl), np.std(session1_bl), np.mean(session2_bl), np.std(session2_bl), stats.ttest_ind(session1_bl, session2_bl)[1]))
        print('& LCD & %.4f (%.4f) & %.4f (%.4f) & %g \\\\ \cline{2-4}' % (np.mean(session1_lcd), np.std(session1_lcd), np.mean(session2_lcd), np.std(session2_lcd), stats.ttest_ind(session1_lcd, session2_lcd)[1]))
        print('& BR & %.4f (%.4f) & %.4f (%.4f) & %g \\\\ \cline{2-4}' % (np.mean(session1_br), np.std(session1_br), np.mean(session1_br), np.std(session1_br), stats.ttest_ind(session1_br, session2_br)[1]))
        print('')

def filter_results(results, interval):
    filtered_results = [scipy.signal.medfilt(data, interval) for data in results[:-1]]
    filtered_results.append(results[-1])
    return filtered_results

def plot_animation_units_for_speaker(id):
    session1_files = get_session_files(1, id)
    session2_files = get_session_files(2, id)

    session1_lr, session1_jl, session1_ls, session1_bl, session1_lcd, session1_br, session1_dates = filter_results(get_au_per_speaker(session1_files),101)

    session2_lr, session2_jl, session2_ls, session2_bl, session2_lcd, session2_br, session2_dates = filter_results(get_au_per_speaker(session2_files),101)

    f, axarr = plt.subplots(6, sharex=True)
    for i,data in enumerate([(session1_lr, session2_lr), (session1_jl, session2_jl), (session1_ls, session2_ls), (session1_bl, session2_bl), (session1_lcd, session2_lcd), (session1_br, session2_br)]):
        session1_data = data[0]
        session2_data = data[1]
        #filtered_data = scipy.signal.medfilt(data, interval+1)
        #relevant_points = get_relevant_intervals(interval, data, num)
        session1_data_line, = axarr[i].plot(range(0, len(session1_data)), session1_data, 'g', label='Session 1')
        session2_data_line, = axarr[i].plot(range(0, len(session2_data)), session2_data, 'y', label='Session 2')
        axarr[i].set_ylim([-1,1])
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()
    #plt.savefig('test.png')

def merge_all_au_data():
    ids = [2,3,4,5,9]
    session1_files = []
    session2_files = []
    for id in ids:
        session1_files += get_session_files(1, id)
        session2_files += get_session_files(2, id)

    session1_lr, session1_jl, session1_ls, session1_bl, session1_lcd, session1_br, session1_dates = filter_results(get_au_per_speaker(session1_files),101)
    session2_lr, session2_jl, session2_ls, session2_bl, session2_lcd, session2_br, session2_dates = filter_results(get_au_per_speaker(session2_files),101)
    print('Session1 | Session 2 | ttest  ')
    print('Lip raiser: mean=%f, std=%f | mean=%f, std=%f | %s' % (np.mean(session1_lr), np.std(session1_lr), np.mean(session2_lr), np.std(session2_lr), stats.ttest_ind(session1_lr, session2_lr)))
    #print(np.correlate(session1_lr, session2_lr))
    print('Jaw lower: mean=%f, std=%f | mean=%f, std=%f | %s' % (np.mean(session1_jl), np.std(session1_jl), np.mean(session2_jl), np.std(session2_jl), stats.ttest_ind(session1_jl, session2_jl)))
    #print(np.correlate(session1_jl, session2_jl))
    print('Lip stretcher: mean=%f, std=%f | mean=%f, std=%f | %s' % (np.mean(session1_ls), np.std(session1_ls), np.mean(session2_ls), np.std(session2_ls), stats.ttest_ind(session1_ls, session2_ls)))
    #print(np.correlate(session1_ls, session2_ls))
    print('Brow lower: mean=%f, std=%f | mean=%f, std=%f | %s' % (np.mean(session1_bl), np.std(session1_bl), np.mean(session2_bl), np.std(session2_bl), stats.ttest_ind(session1_bl, session2_bl)))
    #print(np.correlate(session1_bl, session2_bl))
    print('Lip corner depressor: mean=%f, std=%f | mean=%f, std=%f | %s' % (np.mean(session1_lcd), np.std(session1_lcd), np.mean(session2_lcd), np.std(session2_lcd), stats.ttest_ind(session1_lcd, session2_lcd)))
    #print(np.correlate(session1_lcd, session2_lcd))
    print('Brow raiser: mean=%f, std=%f | mean=%f, std=%f | %s' % (np.mean(session1_br), np.std(session1_br), np.mean(session1_br), np.std(session1_br), stats.ttest_ind(session1_br, session2_br)))
    #print(np.correlate(session1_br, session2_br))
    print('')

# For each participant, extract mean, skewness, kurtosis, max, min, stdev of AUs in S1
def extract_features():

    with open('au_features.csv', 'w') as f:
        csvwriter = csv.writer(f)

        session_ids = [[1,2,3,4,5,6,7,8,9,10,11], [2,3,4,5,9]] 
        for session in [1, 2]:
        #for id in [1,2,3,4,5,6,7,8,9,10,11]:
        #    session_files = get_session_files(1, id)
            for id in session_ids[session-1]:
                session_files = get_session_files(session, id)
                lr, jl, ls, bl, lcd, br, dates = filter_results(get_au_per_speaker(session_files), 101)
            
                data = [lr, jl, ls, bl, lcd, br]
                au_types = ['Lip raiser', 'Jaw lower', 'Lip stretcher', 'Brow lower', 'Lip corner depressor', 'Brow raiser']

                print(id)
                features = []
                for au, au_type in zip(data, au_types):
                    print(au_type)
                    print('%f, %f, %f, %f, %f, %f' % (np.mean(au), stats.skew(au), stats.kurtosis(au), np.max(au), np.min(au), np.std(au)))
                    
                    # Find number of changepoints
                    ts = TimeSeriesAnalysis(dates, au)
                    ts.ChangePointAnalysis()
                    num_changepoints = len(ts.changepoints)

                    features += [np.mean(au), stats.skew(au), stats.kurtosis(au), np.max(au), np.min(au), np.std(au), num_changepoints]
                    # print('Mean: %f' % np.mean(au))
                    # print('Skewness: %f' % stats.skew(au))
                    # print('Kurtosis: %f' % stats.kurtosis(au))
                    # print('Max: %f' % np.max(au))
                    # print('Min: %f' % np.min(au))
                    # print('Stdev: %f' %  np.std(au))
                csvwriter.writerow([session, id] + features)



def analyze(id, interval):
    changepoints_file = '/u/chloe/ludwig/kinect_analysis/changepoints_P%d.csv' % id
    with open(changepoints_file) as f:
        lines = f.readlines()
    lr = [float(el) for el in lines[0].split(',')]
    jl = [float(el) for el in lines[1].split(',')]
    ls = [float(el) for el in lines[2].split(',')]
    bl = [float(el) for el in lines[3].split(',')]
    lcd = [float(el) for el in lines[4].split(',')]
    br = [float(el) for el in lines[5].split(',')]

    changepoints = sorted(lr + jl + ls + bl + lcd + br)
    utterances, actions, utterance_dates, action_dates = get_robot_actions(id)

    robot_actions = utterance_dates + action_dates
    print 'P%d' % id
    print '# robot actions: %d' % len(robot_actions)

    successful_actions = 0
    successful_cp = 0

    for action in robot_actions:
        idx = np.where((np.array(changepoints) >= action) & (np.array(changepoints) <= action + interval))
        relevant_changes = len(idx[0])
        if relevant_changes > 0:
            successful_actions += 1
            successful_cp += relevant_changes

    print '# actions that lead to CP: %d (%f)' % (successful_actions, 1.0* successful_actions/len(robot_actions))
    print '# CP that changed: %d (%f)' % (successful_cp, 1.0* successful_cp/len(changepoints))
