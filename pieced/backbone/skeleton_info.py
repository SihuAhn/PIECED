# NTU RGB+D (25 joints) Bone-pairs
# NTU_BONES = [
#     (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6),
#     (8, 20), (9, 8), (10, 9), (11, 10), (12, 11), (13, 20), (14, 13),
#     (15, 14), (16, 15), (17, 16), (18, 20), (19, 18),
#     (21, 19), (22, 21), (23, 22), (24, 23)
# ]
NTU_BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24), (0, 12), (12, 13), (13, 14), 
    (14, 15), (0, 16), (16, 17), (17, 18), (18, 19)
]

# Kinetics (COCO 18 joints) Bone-pairs
KINETICS_BONES = [
    # Torso (Center)
    (0, 1), (1, 2), (1, 5), (1, 8), (1, 11),    # Nose -> Neck # Neck -> R-Shoulder # Neck -> L-Shoulder # Neck -> R-Hip # Neck -> L-Hip
    # Right Arm
    (2, 3),   # R-Shoulder -> R-Elbow
    (3, 4),   # R-Elbow -> R-Wrist
    
    # Left Arm
    (5, 6),   # L-Shoulder -> L-Elbow
    (6, 7),   # L-Elbow -> L-Wrist
    
    # Right Leg
    (8, 9),   # R-Hip -> R-Knee
    (9, 10),  # R-Knee -> R-Ankle
    
    # Left Leg
    (11, 12), # L-Hip -> L-Knee
    (12, 13), # L-Knee -> L-Ankle
    
    # Face (Right)
    (0, 14),  # Nose -> R-Eye
    (14, 16), # R-Eye -> R-Ear
    
    # Face (Left)
    (0, 15),  # Nose -> L-Eye
    (15, 17)  # L-Eye -> L-Ear
]

# 데이터셋 이름에 따른 bone 정보 딕셔너리
BONES_INFO = {
    'ntu60': NTU_BONES,
    'ntu120': NTU_BONES,
    'kinetics400': KINETICS_BONES,
    'pkuv1': NTU_BONES,
    'pkuv2': NTU_BONES
}