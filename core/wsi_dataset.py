import openslide
from torch.utils.data import Dataset

class SingleWSIDataset(Dataset):
    """将单个 WSI 按 grid 坐标拆分成多个 patch 的 Dataset"""
    
    def __init__(self, grid, slide_path, transform=None):
        self.grid = grid
        self.slide = openslide.open_slide(slide_path)
        self.transform = transform
        
        # 获取放大倍数
        if 'mirax.GENERAL.OBJECTIVE_MAGNIFICATION' in self.slide.properties:
            self.mag = int(self.slide.properties['mirax.GENERAL.OBJECTIVE_MAGNIFICATION'])
        elif 'aperio.AppMag' in self.slide.properties:
            self.mag = int(self.slide.properties['aperio.AppMag'])
        else:
            self.mag = 20
            print('==> Cannot Find WSI MAGNIFICATION Parameter, SET mag=20!')
        
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