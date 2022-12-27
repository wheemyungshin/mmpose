dataset_info = dict(
    dataset_name='cocoupperbody',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        2:
        dict(
            name='right_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        3:
        dict(
            name='left_ear',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        4:
        dict(
            name='right_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
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
        dict(
            link=('right_shoulder', 'right_elbow'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('left_elbow', 'left_wrist'), id=3, color=[0, 255, 0]),
        4:
        dict(link=('right_elbow', 'right_wrist'), id=4, color=[255, 128, 0]),
        5:
        dict(link=('left_eye', 'right_eye'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('nose', 'left_eye'), id=6, color=[51, 153, 255]),
        7:
        dict(link=('nose', 'right_eye'), id=7, color=[51, 153, 255]),
        8:
        dict(link=('left_eye', 'left_ear'), id=8, color=[51, 153, 255]),
        9:
        dict(link=('right_eye', 'right_ear'), id=9, color=[51, 153, 255]),
        10:
        dict(link=('left_ear', 'left_shoulder'), id=10, color=[51, 153, 255]),
        11:
        dict(
            link=('right_ear', 'right_shoulder'), id=11, color=[51, 153, 255])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062
    ])
