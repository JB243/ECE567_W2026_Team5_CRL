import argparse
import numpy as np
from continual_rl.utils.metrics import Metrics

# see https://github.com/plotly/Kaleido/issues/101
import plotly.io as pio
pio.kaleido.scope.mathjax = None  # Prevents a weird "Loading MathJax" artifact in rendering the pdf


TASKS_ATARI = {
    "0-SpaceInvaders": dict(i=0, y_range=[0, 4e3], yaxis_dtick=1e3, train_regions=[[0, 50e6], [300e6, 350e6]], showlegend=False),
    "1-Krull": dict(i=1, y_range=[0, 1e4], yaxis_dtick=2e3, train_regions=[[50e6, 100e6], [350e6, 400e6]], showlegend=False),
    "2-BeamRider": dict(i=2, y_range=[0, 1e4], yaxis_dtick=2e3, train_regions=[[100e6, 150e6], [400e6, 450e6]], showlegend=True),
    "3-Hero": dict(i=3, y_range=[0, 5e4], yaxis_dtick=1e4, train_regions=[[150e6, 200e6], [450e6, 500e6]], showlegend=False),
    "4-StarGunner": dict(i=4, y_range=[0, 10e4], yaxis_dtick=2e4, train_regions=[[200e6, 250e6], [500e6, 550e6]], showlegend=False),
    "5-MsPacman": dict(i=5, y_range=[0, 4e3], yaxis_dtick=1e3, train_regions=[[250e6, 300e6], [550e6, 600e6]], showlegend=True),
}


MODELS_ATARI = {
    "IMPALA": dict(
        name='impala',
        runs=[f'impala{i}' for i in range(5)],
        # color='rgba(64, 132, 133, 1)',
        color='rgba(77, 102, 133, 1)',
        color_alpha=0.2,
    ),
    "EWC": dict(
        name='ewc',
        runs=[f'ewc{i}' for i in range(5)],
        color='rgba(214, 178, 84, 1)',
        color_alpha=0.2,
    ),
    "ONLINE EWC": dict(
        name='online ewc',
        runs=[f'onlineewc{i}' for i in range(5)],
        color='rgba(106, 166, 110, 1)',
        color_alpha=0.2,
    ),
    "P&C": dict(
        name='pnc',
        runs=['pnc0', 'pnc1', 'pnc2', 'pnc3_last1Mlost', 'pnc4'],
        # color='rgba(152, 52, 48, 1)',
        color='rgba(152, 67, 63, 1)',
        color_alpha=0.2,
    ),
    "CLEAR": dict(
        name='clear',
        runs=['clear0', 'clear1', 'clear2', 'clear5', 'clear8'],
        # color='rgba(212, 162, 217, 1)',
        color='rgba(210, 140, 217, 1)',
        color_alpha=0.2,
    ),
}
ATARI = dict(
    models=MODELS_ATARI,
    tasks=TASKS_ATARI,
    num_cycles=2,
    num_cycles_for_forgetting=1,
    num_task_steps=50e6,
    grid_size=[2, 3],
    which_exp='atari',
    rolling_mean_count=20,
    filter='ma',
    xaxis_tickvals=list(np.arange(0, 600e6 + 1, 300e6)),
    cache_dir='tmp/cache/data_pkls/atari/',
)


TASKS_PROCGEN = {
    "0-Climber": dict(i=0, eval_i=1, y_range=[0., 1.25], yaxis_dtick=0.25, train_regions=[[5e6 * i, 5e6 * (i + 1)] for i in range(0, 6 * 5, 6)]),
    "1-Dodgeball": dict(i=2, eval_i=3, y_range=[0., 3.], yaxis_dtick=0.5, train_regions=[[5e6 * i, 5e6 * (i + 1)] for i in range(1, 6 * 5, 6)]),
    "2-Ninja": dict(i=4, eval_i=5, y_range=[0., 5.], yaxis_dtick=1.0, train_regions=[[5e6 * i, 5e6 * (i + 1)] for i in range(2, 6 * 5, 6)]),
    "3-Starpilot": dict(i=6, eval_i=7, y_range=[0., 55.], yaxis_dtick=5.0, train_regions=[[5e6 * i, 5e6 * (i + 1)] for i in range(3, 6 * 5, 6)]),
    "4-Bigfish": dict(i=8, eval_i=9, y_range=[0., 18.], yaxis_dtick=3.0, train_regions=[[5e6 * i, 5e6 * (i + 1)] for i in range(4, 6 * 5, 6)]),
    "5-Fruitbot": dict(i=10, eval_i=11, y_range=[-3, 30], yaxis_dtick=5, train_regions=[[5e6 * i, 5e6 * (i + 1)] for i in range(5, 6 * 5, 6)]),
}
MODELS_PROCGEN = {
    "IMPALA": dict(
        name='impala',
        runs=['impala0_procgen/impala_procgen/0'],
        color='rgba(77, 102, 133, 1)',
        color_alpha=0.2,
    ),
    "EWC": dict(
        name='ewc',
        runs=['tmp_procgen_ewc/ewc_procgen/0'],
        color='rgba(214, 178, 84, 1)',
        color_alpha=0.2,
    ),
    "ONLINE EWC": dict(
        name='online ewc',
        runs=['tmp_procgen_online_ewc/online_ewc_procgen/0'],
        color='rgba(106, 166, 110, 1)',
        color_alpha=0.2,
    ),
    "P&C": dict(
        name='pnc',
        runs=['tmp_procgen_pnc/pnc_procgen/0'],
        color='rgba(152, 67, 63, 1)',
        color_alpha=0.2,
    ),
    "CLEAR": dict(
        name='clear',
        runs=['clear0_procgen/clear_procgen/0'],
        color='rgba(210, 140, 217, 1)',
        color_alpha=0.2,
    ),
}
PROCGEN = dict(
    models=MODELS_PROCGEN,
    tasks=TASKS_PROCGEN,
    rolling_mean_count=20,
    filter='ma',
    num_cycles=5,
    num_cycles_for_forgetting=1,
    num_task_steps=5e6,
    grid_size=[2, 3],
    which_exp='procgen',
    xaxis_tickvals=list(np.arange(0, 150e6 + 1, 25e6)),
    cache_dir='tmp',
)


TASKS_MINIHACK = {
    "0-Room-Random": dict(i=0, eval_i=1, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(0, 15 * 2, 15)]),
    "1-Room-Dark": dict(i=2, eval_i=3, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(1, 15 * 2, 15)]),
    "2-Room-Monster": dict(i=4, eval_i=5, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(2, 15 * 2, 15)]),
    "3-Room-Trap": dict(i=6, eval_i=7, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(3, 15 * 2, 15)]),
    "4-Room-Ultimate": dict(i=8, eval_i=9, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(4, 15 * 2, 15)]),
    "5-Corridor-R2": dict(i=10, eval_i=11, y_range=[-1, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(5, 15 * 2, 15)]),
    "6-Corridor-R3": dict(i=12, eval_i=13, y_range=[-1, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(6, 15 * 2, 15)]),
    "7-KeyRoom": dict(i=14, eval_i=15, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(7, 15 * 2, 15)]),
    "8-KeyRoom-Dark": dict(i=16, eval_i=17, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(8, 15 * 2, 15)]),
    "9-River-Narrow": dict(i=18, eval_i=19, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(9, 15 * 2, 15)]),
    "10-River-Monster": dict(i=20, eval_i=21, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(10, 15 * 2, 15)]),
    "11-River-Lava": dict(i=22, eval_i=23, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(11, 15 * 2, 15)]),
    "12-HideNSeek": dict(i=24, eval_i=25, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(12, 15 * 2, 15)]),
    "13-HideNSeek-Lava": dict(i=26, eval_i=27, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(13, 15 * 2, 15)]),
    "14-CorridorBattle": dict(i=28, eval_i=29, y_range=[-0.5, 1.], train_regions=[[10e6 * i, 10e6 * (i + 1)] for i in range(14, 15 * 2, 15)]),
}


impala_minihack_paths = ['impala1_minihack', 'impala2_minihack', 'henry_runs/impala_minihack_paperdefaults/0', 'henry_runs/impala_minihack_paperdefaults/1', 'henry_runs/impala_minihack_paperdefaults/2']
clear_minihack_paths = ['clear0_minihack', 'clear1_minihack', 'clear2_minihack', 'clear3_minihack', 'clear4_minihack']
MODELS_MINIHACK = {
    "IMPALA": dict(
        name='impala',
        runs=impala_minihack_paths,
        color='rgba(100, 149, 237, 1)',
        color_alpha=0.2,
    ),
    "CLEAR": dict(
        name='clear',
        runs=clear_minihack_paths,
        # color='rgba(212, 162, 217, 1)',
        color='rgba(210, 140, 217, 1)',
        color_alpha=0.2,
    ),
}
MINIHACK = dict(
    models=MODELS_MINIHACK,
    tasks=TASKS_MINIHACK,
    rolling_mean_count=20,
    filter='ma',
    num_cycles=2,
    num_cycles_for_forgetting=1,
    num_task_steps=10e6,
    which_exp='minihack',
    xaxis_tickvals=list(np.arange(0, 260e6 + 1, 130e6)),
    metric_eps=0.1,
    cache_dir='tmp/cache/data_pkls/minihack/',
)


TASKS_CHORE_VARY_ENV = {
    "R402": dict(i=0, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(0, 6, 3)]),
    "R419": dict(i=1, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(1, 6, 3)]),
    "R423": dict(i=2, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(2, 6, 3)])
}
MODELS_CHORE_VARY_ENV = {
    "EWC": dict(
        name='ewc',
        runs=[f'vary_envs_2/{i}' for i in [3, 4, 5]],
        color='rgba(214, 178, 84, 1)',
        color_alpha=0.2,
    ),
    "P&C": dict(
        name='pnc',
        runs=[f'vary_envs_2/{i}' for i in [9, 10, 11]],
        color='rgba(152, 67, 63, 1)',
        color_alpha=0.2,
    ),
    "CLEAR": dict(
        name='clear',
        runs=[f'vary_envs_2/{i}' for i in [0, 1, 2]],
        color='rgba(210, 140, 217, 1)',
        color_alpha=0.2,
    ),
}
CHORE_VARY_ENV = dict(
    models=MODELS_CHORE_VARY_ENV,
    tasks=TASKS_CHORE_VARY_ENV,
    rolling_mean_count=5,
    filter='ma',
    num_cycles=2,
    num_cycles_for_forgetting=1,
    num_task_steps=1e6,
    which_exp='chore_vary_env',
    clip_y_range=[-10, 12],
    cache_dir='tmp/cache/data_pkls/chores/',
)


TASKS_CHORE_VARY_TASK = {
    "Hang TP": dict(i=0, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(0, 6, 3)]),
    "Counter": dict(i=1, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(1, 6, 3)]),
    "Cabinet": dict(i=2, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(2, 6, 3)])
}

MODELS_CHORE_VARY_TASK = {
    "EWC": dict(
        name='ewc',
        runs=[f'vary_tasks_2/{i}' for i in [3, 4, 5]],
        color='rgba(214, 178, 84, 1)',
        color_alpha=0.2,
    ),
    "P&C": dict(
        name='pnc',
        runs=[f'vary_tasks_2/{i}' for i in [6, 7, 8]],
        color='rgba(152, 67, 63, 1)',
        color_alpha=0.2,
    ),
    "CLEAR": dict(
        name='clear',
        runs=[f'vary_tasks_2/{i}' for i in [0, 1, 2]],
        color='rgba(210, 140, 217, 1)',
        color_alpha=0.2,
    ),
}
CHORE_VARY_TASK = dict(
    models=MODELS_CHORE_VARY_TASK,
    tasks=TASKS_CHORE_VARY_TASK,
    rolling_mean_count=5,
    filter='ma',
    num_cycles=2,
    num_cycles_for_forgetting=1,
    num_task_steps=1e6,
    which_exp='chore_vary_task',
    clip_y_range=[-10, None],
    cache_dir='tmp/cache/data_pkls/chores/'
)


TASKS_CHORE_VARY_OBJECT = {
    "Fork": dict(i=0, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(0, 6, 3)]),
    "Knife": dict(i=1, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(1, 6, 3)]),
    "Spoon": dict(i=2, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(2, 6, 3)])
}
MODELS_CHORE_VARY_OBJECT = {
    "EWC": dict(
        name='ewc',
        runs=[f'vary_objects_3/{i}' for i in [3, 4, 5]],
        color='rgba(214, 178, 84, 1)',
        color_alpha=0.2,
    ),
    "P&C": dict(
        name='pnc',
        runs=[f'vary_objects_3/{i}' for i in [6, 7, 8]],
        color='rgba(152, 67, 63, 1)',
        color_alpha=0.2,
    ),
    "CLEAR": dict(
        name='clear',
        runs=[f'vary_objects_3/{i}' for i in [0, 1, 2]],
        color='rgba(210, 140, 217, 1)',
        color_alpha=0.2,
    ),
}
CHORE_VARY_OBJECT = dict(
    models=MODELS_CHORE_VARY_OBJECT,
    tasks=TASKS_CHORE_VARY_OBJECT,
    rolling_mean_count=5,
    filter='ma',
    num_cycles=1,
    num_cycles_for_forgetting=1,
    num_task_steps=1e6,
    which_exp='chore_vary_object',
    clip_y_range=[-10, None],
    cache_dir='tmp/cache/data_pkls/chores/'
)


TASKS_CHORE_MULTI_TRAJ = {
    "R19, Cup": dict(i=0, eval_i=1, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(0, 6, 3)]),
    "R13, Potato": dict(i=2, eval_i=3, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(1, 6, 3)]),
    "R02, Lettuce": dict(i=4, eval_i=5, y_range=[-15, 15.], train_regions=[[1e6 * i, 1e6 * (i + 1)] for i in range(2, 6, 3)])
}
MODELS_CHORE_MULTI_TRAJ = {
    "EWC": dict(
        name='ewc',
        runs=[f'multi_traj/{i}' for i in [3, 4, 5]],
        color='rgba(214, 178, 84, 1)',
        color_alpha=0.2,
    ),
    "P&C": dict(
        name='pnc',
        runs=[f'multi_traj/{i}' for i in [6, 7, 8]],
        color='rgba(152, 67, 63, 1)',
        color_alpha=0.2,
    ),
    "CLEAR": dict(
        name='clear',
        runs=[f'multi_traj/{i}' for i in [0, 1, 2]],
        color='rgba(210, 140, 217, 1)',
        color_alpha=0.2,
    ),
}
CHORE_MULTI_TRAJ = dict(
    models=MODELS_CHORE_MULTI_TRAJ,
    tasks=TASKS_CHORE_MULTI_TRAJ,
    rolling_mean_count=5,
    filter='ma',
    num_cycles=1,
    num_cycles_for_forgetting=1,
    num_task_steps=1e6,
    which_exp='chore_multi_traj',
    clip_y_range=[-10, None],
    cache_dir='tmp/cache/data_pkls/chores/'
)


TO_PLOT = dict(
    tag_base='eval_reward',
    cache_dir='tmp/',
    legend_size=30,
    title_size=40,
    axis_size=20,
    axis_label_size=30,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help='experiment dir')
    args = parser.parse_args()
    TO_PLOT['exp_dir'] = args.d

    #exp_data = ATARI
    #exp_data = PROCGEN
    exp_data = MINIHACK
    #exp_data = CHORE_VARY_ENV
    #exp_data = CHORE_VARY_TASK
    #exp_data = CHORE_VARY_OBJECT
    #exp_data = CHORE_MULTI_TRAJ
    TO_PLOT.update(**exp_data)

    metrics = Metrics(TO_PLOT)
    metrics.visualize()
