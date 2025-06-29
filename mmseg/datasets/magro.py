from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

classes = ('road', 'vegetation', 'sky', 'obstacle', 'others')
palette = [[128, 64, 128], [0, 128, 0], [70, 130, 180], [220, 20, 60], [128, 128, 128]]

@DATASETS.register_module()
class MAgroDataset(BaseSegDataset):

    METAINFO = dict(
        classes=classes,
        palette=palette)

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', reduce_zero_label=False, **kwargs)
