import sys
sys.path.append('D:\\depth_sort\\deep_sort_realtime_depth')

import matplotlib.pyplot as plt
from deep_sort_realtime.deepsort_tracker import DeepSort
import warnings
warnings.filterwarnings('ignore')

tracker = DeepSort(
    max_age=1, nms_max_overlap=1.0, max_cosine_distance=0.2
)

bbs = [
    ([136, 520, 51, 135, 297], 1, 1),
    ([[138, 521, 50, 134, 299], 1, 1]),
    ([[ 140, 521, 48, 133, 300], 1, 1]),
    ([[142, 521, 47, 132, 300], 1, 1]),
]
frames = [plt.imread("D:\\depth_sort\\deep_sort_realtime_depth\\img0.jpg"),
          plt.imread("D:\\depth_sort\\deep_sort_realtime_depth\\img1.jpg"),
          plt.imread("D:\\depth_sort\\deep_sort_realtime_depth\\img2.jpg"),
          plt.imread("D:\\depth_sort\\deep_sort_realtime_depth\\img3.jpg"),
]
for i in range(len(frames)):
    frame = frames[i]
    tracks = tracker.update_tracks(bbs, frame=frame)