import cv2
import numpy as np
import time
import math

def drone_position(middle,drone_center,lider,drone_direction, az_radian):
    # 실제 환경 계산 code
    print()
    # 드론중심으로 타겟의 정보
    In_map_target_dy = drone_center[1]
    In_map_target_st = math.sqrt(lider ** 2 - In_map_target_dy ** 2)
    In_map_target_az = (drone_direction * (math.pi/180)) - math.atan2(lider, In_map_target_dy) + az_radian
    print("드론 타겟 사이 꺽인 (라디안)각도 : ", In_map_target_az)
    print("드론 타겟 사이 (일반)각도 : ", math.degrees(In_map_target_az))
    print("실제 드론과 타겟의 거리  : ", In_map_target_st)
    print()
    # 맵 중심의 타켓 정보
    print("맵상 드론이 바라보는 각도 : ", drone_direction)
    # 드론 정보
    print("드론 위치 : ", drone_center)
    # xy : 맵상 좌표 / z: 높이
    In_map_target_cx = drone_center[0] + In_map_target_st * math.cos(math.radians(drone_direction) + In_map_target_az)
    In_map_target_cy = drone_center[1] + In_map_target_st * math.sin(math.radians(drone_direction) + In_map_target_az)
    In_map_target_cz = drone_center[2]
    In_map_target = (In_map_target_cx, In_map_target_cy, In_map_target_cz)
    In_map_target_c = In_map_target
    print("타켓 예상 위치 : ", In_map_target_c)