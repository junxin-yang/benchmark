import openslide
from torch.utils.data import Dataset

class SingleWSIDataset(Dataset):
    """将单个 WSI 按 grid 坐标拆分成多个 patch 的 Dataset"""
    
    def __init__(self, grid, slide_path, transform=None):
        self.grid = grid
        self.slide = openslide.open_slide(slide_path)
        self.transform = transform
        

        try:
            self.mpp = (float(self.slide.properties['openslide.mpp-x']) + float(self.slide.properties['openslide.mpp-y'])) / 2.0
        except KeyError:
            print(f"Warning: MPP information not found in slide properties for {slide_path}. Setting default mpp to 0.25")
            self.mpp = 0.25
        self.thresholds = {
            40.0: 0.25,  # μm/pixel
            20.0: 0.50,
            10.0: 1.00,
        }
        est_mag  = 40 * (0.25 / self.mpp)  # assume 40x is 0.25 mpp
        self.mag = min(self.thresholds.keys(), key=lambda k: abs(est_mag - k))

        
        # 根据放大倍数选择读取层级
        if self.mag == 20:
            self.read_level = 0
        elif self.mag == 40:
            self.read_level = 1
        else:
            raise ValueError(f'Not Support Magnification: {self.mag}')

    def __getitem__(self, index):
        coord = self.grid[index]
        coord = (int(coord[0]), int(coord[1]))
        img = self.slide.read_region(coord, self.read_level, (256, 256)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.grid)
    
    def close_slide(self):
        """关闭 slide 文件句柄"""
        self.slide.close()