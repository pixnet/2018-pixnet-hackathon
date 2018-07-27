import numpy as np


class MaskGenerator:
    def __init__(self, mask_size=(256, 256), box_size=(128, 128)):
        self.mask_size = mask_size
        self.box_size = box_size

    def _draw(self):
        raise NotImplementedError

    def flow(self, batch_size=32, total_size=-1, **draw_kwargs):
        residual = total_size
        below_zero = total_size <= 0
        while below_zero or residual > 0:
            n = batch_size if below_zero else min(batch_size, residual)
            samples = [self._draw(**draw_kwargs) for _ in range(n)]
            masks = np.asarray([mask for mask, box in samples])
            boxes = np.asarray([box for mask, box in samples])
            yield (masks, boxes)
            residual = total_size if below_zero else total_size - n


class FixMaskGenerator(MaskGenerator):

    def _draw(self, x=None, y=None, w=None, h=None):
        box = np.asarray([x, y, self.box_size[1], self.box_size[0]], dtype=np.int)
        p = x + ((self.box_size[1] - w) >> 1)
        q = y + ((self.box_size[0] - h) >> 1)
        mask = np.zeros(self.mask_size + (1,), dtype=np.float)
        mask[q:q+h, p:p+w, 0] = 1.0
        return (mask, box)


class RandomMaskGenerator(MaskGenerator):

    def __init__(self, max_size=(96, 96), min_size=(16, 16), **kwargs):
        super().__init__(**kwargs)
        self.max_size = max_size
        self.min_size = min_size

    def _draw(self, seed=None):
        y = np.random.randint(0, self.mask_size[0] - self.box_size[0] + 1)
        x = np.random.randint(0, self.mask_size[1] - self.box_size[1] + 1)
        box = np.asarray([x, y, self.box_size[1], self.box_size[0]], dtype=np.int)

        w = np.random.randint(self.min_size[1], self.max_size[1] + 1)
        h = np.random.randint(self.min_size[0], self.max_size[0] + 1)

        p = x + ((self.box_size[1] - w) >> 1)
        q = y + ((self.box_size[0] - h) >> 1)

        mask = np.zeros(self.mask_size + (1,), dtype=np.float)
        mask[q:q+h, p:p+w, 0] = 1.0

        return (mask, box)
