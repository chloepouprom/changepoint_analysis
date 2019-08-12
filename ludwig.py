import re
from datetime import datetime
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

def extract_shape_units(filename):
    with open(filename) as f:
        td = f.readlines()

    su_dates = []
    su = np.zeros((3, 121, len(td)))
    su_id = np.zeros((121, len(td)))

    for d, line in enumerate(td):
        data = line.split(';')

        pts = data[:-1]
        date = re.search('(.*)\r\n',data[-1]).groups()[0]

        su_dates.append((datetime.strptime(date, '%Y-%m-DD_%H:%M:%S:%f') - datetime(1970,1,1)).total_seconds())

        print(d)
        for i,pt in enumerate(pts):
            [id,x_coord, y_coord, z_coord] = re.search('ID: ([0-9]+), (.*) (.*) (.*) ', pt).groups()
            print(pt)
            su[0,i,:] = x_coord
            su[1,i,:] = y_coord
            su[2,i,:] = z_coord
            su_id[i, d] = id
        
    return [su, su_id, su_dates]

def plot_shape_units(su, filename):
    plt.clf()
    for i in range(121):
        plt.plot(su[0,i,:], su[2,i,:])
        # plt.plot(su[0,i,:], su[1,i,:], 'o')

    plt.savefig(filename)

# au_file = '/p/spoclab/data/Ludwig/ROBOT_DATA/Session2/2/AU_session-2_id-2_order-1_0.txt'
# [lip_raiser_data, jaw_lower_data, lip_stretcher_data, brow_lower_data, lip_corner_depressor_data, brow_raiser_data, au_dates] = ludwig.extract_animation_units(au_file)
def extract_animation_units(filename):

    lip_raiser_data = []
    jaw_lower_data = []
    lip_stretcher_data = []
    brow_lower_data = []
    lip_corner_depressor_data = []
    brow_raiser_data = []
    au_dates = []

    with open(filename) as f:
        au = f.readlines()


    for line in au:
        [id, lip_raiser, jaw_lower, lip_stretcher, brow_lower, lip_corner_depressor, brow_raiser, date]= re.search('ID: ([0-9]+), (.*) (.*) (.*) (.*) (.*) (.*) (.*)', line).groups()

        lip_raiser_data.append(float(lip_raiser))
        jaw_lower_data.append(float(jaw_lower))
        lip_stretcher_data.append(float(lip_stretcher))
        brow_lower_data.append(float(brow_lower))
        lip_corner_depressor_data.append(float(lip_corner_depressor))
        brow_raiser_data.append(float(brow_raiser))

        au_dates.append((datetime.strptime(date, '%Y-%m-DD_%H:%M:%S:%f') - datetime(1970,1,1)).total_seconds())

    return [lip_raiser_data, jaw_lower_data, lip_stretcher_data, brow_lower_data, lip_corner_depressor_data, brow_raiser_data, au_dates]


# robot_file = '/p/spoclab/data/Ludwig/ROBOT_DATA/Session2/2/RobotActions_session-2_id-2_order-1_0.txt'
# [utterances, actions, utterance_dates, action_dates] = ludwig.extract_robot_actions(robot_file)
def extract_robot_actions(filename):

    utterances = []
    actions = []
    utterance_dates = []
    action_dates = []

    with open(filename) as f:
        robot = f.readlines()

    for line in robot:
        [utterance, action, date] = re.search('Utterance: (.*), Action: (.*),(.*)\r\n', line).groups()

        if action != 'NULL':
            actions.append(action)
            action_dates.append((datetime.strptime(date, '%Y-%m-DD_%H:%M:%S:%f') - datetime(1970,1,1)).total_seconds())

        if utterance != 'NULL':
            utterances.append(utterance)
            utterance_dates.append((datetime.strptime(date, '%Y-%m-DD_%H:%M:%S:%f') - datetime(1970,1,1)).total_seconds())

    return [utterances, actions, utterance_dates, action_dates]

# ludwig.plot_au_with_actions(lip_raiser_data, jaw_lower_data, lip_stretcher_data, brow_lower_data, lip_corner_depressor_data, brow_raiser_data, au_dates,utterances, actions, utterance_dates, action_dates, 'test.png')
def plot_au_with_actions(lr, jl, ls, bl, lcd, br, au_dates, utt, act, utt_dates, act_dates, filename):
    f, axarr = plt.subplots(6, sharex=True)
    axarr[0].plot(au_dates, lr, 'y', zorder=1)
    axarr[0].set_title('Lip raiser')
    axarr[0].set_ylim([-1,1])
    axarr[1].plot(au_dates, jl, 'y', zorder=1)
    axarr[1].set_title('Jaw lower')
    axarr[1].set_ylim([-1,1])
    axarr[2].plot(au_dates, ls, 'y', zorder=1)
    axarr[2].set_title('Lip stretcher')
    axarr[2].set_ylim([-1,1])
    axarr[3].plot(au_dates, bl, 'y', zorder=1)
    axarr[3].set_title('Brow lower')
    axarr[3].set_ylim([-1,1])
    axarr[4].plot(au_dates, lcd, 'y', zorder=1)
    axarr[4].set_title('Lip corner depressor')
    axarr[4].set_ylim([-1,1])
    axarr[5].plot(au_dates, br, 'y', zorder=1)
    axarr[5].set_title('Brow raiser')
    axarr[5].set_ylim([-1,1])

    for utt in utt_dates:
        for i in range(6):
            axarr[i].axvline(x=utt, color='m', linewidth=2, clip_on=False, zorder=0)

    for act in act_dates:
        for i in range(6):
            axarr[i].axvline(x=act, color='m', linewidth=2, clip_on=False, zorder=0)

    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    f.subplots_adjust(hspace=0.3)
    f.savefig(filename)

def plot_au(lr, jl, ls, bl, lcd, br, au_dates, filename):
    f, axarr = plt.subplots(6, sharex=True)
    axarr[0].plot(au_dates, lr, 'y')
    axarr[0].set_title('Lip raiser')
    axarr[0].set_ylim([-1,1])
    axarr[1].plot(au_dates, jl, 'y')
    axarr[1].set_title('Jaw lower')
    axarr[1].set_ylim([-1,1])
    axarr[2].plot(au_dates, ls, 'y')
    axarr[2].set_title('Lip stretcher')
    axarr[2].set_ylim([-1,1])
    axarr[3].plot(au_dates, bl, 'y')
    axarr[3].set_title('Brow lower')
    axarr[3].set_ylim([-1,1])
    axarr[4].plot(au_dates, lcd, 'y')
    axarr[4].set_title('Lip corner depressor')
    axarr[4].set_ylim([-1,1])
    axarr[5].plot(au_dates, br, 'y')
    axarr[5].set_title('Brow raiser')
    axarr[5].set_ylim([-1,1])
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    f.subplots_adjust(hspace=0.3)
    f.savefig(filename)

# session_directory = '/p/spoclab/data/Ludwig/ROBOT_DATA/Session2/'
def plot_au_and_actions_session2(session_directory):
    for id in os.listdir(session_directory):
        au_files = glob.glob(session_directory + str(id) + '/AU_session*')
        robot_files = glob.glob(session_directory + str(id) + '/RobotActions*')

        for au_file, robot_file in zip(au_files, robot_files):
            filename = re.search('(AU_.*)\.txt', au_file).groups()[0] + '.png'
            [lip_raiser_data, jaw_lower_data, lip_stretcher_data, brow_lower_data, lip_corner_depressor_data, brow_raiser_data, au_dates] = extract_animation_units(au_file)
            [utterances, actions, utterance_dates, action_dates] = extract_robot_actions(robot_file)
            plot_au_with_actions(lip_raiser_data, jaw_lower_data, lip_stretcher_data, brow_lower_data, lip_corner_depressor_data, brow_raiser_data, au_dates,utterances, actions, utterance_dates, action_dates, filename)

# session_directory = '/p/spoclab/data/Ludwig/ROBOT_DATA/Session1/'
def plot_au_session1(session_directory):
    for id in ['1', '11_1', '3', '5', '7', '9', '10_1', '2', '4', '6', '8']:
        au_files = glob.glob(session_directory + str(id) + '/AU_session*')

        for au_file in au_files:
            filename = re.search('(AU_.*)\.txt', au_file).groups()[0] + '.png'
            [lip_raiser_data, jaw_lower_data, lip_stretcher_data, brow_lower_data, lip_corner_depressor_data, brow_raiser_data, au_dates] = extract_animation_units(au_file)
            plot_au(lip_raiser_data, jaw_lower_data, lip_stretcher_data, brow_lower_data, lip_corner_depressor_data, brow_raiser_data, au_dates, filename)

def plot_su_session2(session_directory):
    for id in os.listdir(session_directory):
        su_files = glob.glob(session_directory + str(id) + '/TD_session*')

        for su_file in su_files:
            filename = re.search('(TD_.*)\.txt', su_file).groups()[0] + '.png'
            [su, su_dates] = extract_shape_units(su_file)
            plot_shape_units(su, filename)

def plot_su_session1(session_directory):
    for id in ['1', '11_1', '3', '5', '7', '9', '10_1', '2', '4', '6', '8']:
        su_files = glob.glob(session_directory + str(id) + '/TD_session*')
    
        for su_file in su_files:
            filename = re.search('(TD_.*)\.txt', su_file).groups()[0] + '.png'
            [su, su_dates] = extract_shape_units(su_file)
            plot_shape_units(su, filename)

def classify_robot_actions(robot_actions):
    actions = [];

    ACTION_WAVE = '/usr/robokind/etc/gui/anims/AZR25_handWave_01_5000.anim.xml'
    ACTION_HAPPY = '/mnt/alternate/milo/animations/misc-demo-factory/happy_02.rkanim'
    ACTION_LOOK_RIGHT = '/usr/robokind/etc/gui/anims/AZR25_LookRightShoulder_01.anim.xml'

    for action in robot_actions:
        if action == ACTION_WAVE:
            actions.append('WAVE')
        elif action == ACTION_HAPPY:
            actions.append('HAPPY')
        elif action == ACTION_LOOK_RIGHT:
            actions.append('LOOK_RIGHT')
    return actions


