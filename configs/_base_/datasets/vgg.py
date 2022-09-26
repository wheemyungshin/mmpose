dataset_info = dict(
    dataset_name='vgg',
    paper_info=dict(),
    keypoint_info={
        0:
        dict(name='head', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='right_wrist',
            id=1,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        2:
        dict(
            name='left_wrist',
            id=2,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        3:
        dict(
            name='right_elbow',
            id=3,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        4:
        dict(
            name='left_elbow',
            id=4,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        5:
        dict(
            name='right_shoulder',
            id=5,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        6:
        dict(
            name='left_shoulder',
            id=6,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder')
    },
    skeleton_info={
        0:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=0,
            color=[51, 153, 255]),
        1:
        dict(link=('left_shoulder', 'left_elbow'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_shoulder', 'right_elbow'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('left_elbow', 'left_wrist'), id=3, color=[0, 255, 0]),
        4:
        dict(link=('right_elbow', 'right_wrist'), id=4, color=[255, 128, 0]),
    },
    joint_weights=[
        1.5, 1.5, 1.5, 1.2, 1.2, 1., 1.
    ],#1.5 weight for head is temporary set
    sigmas=[
        0.062, 0.062, 0.062, 0.072, 0.072, 0.079, 0.079
    ])