import open3d as o3d
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import random
from torch.nn import CosineSimilarity

class Point:
    def __init__(self, id, coord_2d, coord_3d, feature):
        self.id = id
        self.coord_2d = coord_2d # (frame_id, i, j)
        self.coord_3d = coord_3d # (x, y, z)
        self.feature = feature
        self.length = 1
    
    def update(self, point):
        self.coord_3d = (self.coord_3d*self.length + point.coord_3d*point.length) / (self.length + point.length) # weighted average
        self.feature = (self.feature*self.length + point.feature*point.length) / (self.length + point.length) # weighted average
        self.length += point.length
    
class Compressor:
    def __init__(self, points=defaultdict(Point), max_frame=12):
        self.points = points # this is a dict with key as point id and value as Point object
        self.id_pair = []
        self.sim_score = []
        self.max_frame = max_frame
        self.cos = CosineSimilarity(dim=1)
    
    def __len__(self):
        return len(self.points)
    
    def __repr__(self):
        return f'3D Point Compressor with {len(self)} points'

    def similarity(self, pts1, pts2):
        feats1 = torch.stack([pt.feature for pt in pts1])
        feats2 = torch.stack([pt.feature for pt in pts2])
        return self.cos(feats1, feats2).tolist()
    
    def compress(self, voxel_size=0.1, r=0.3):
        
        # stage 0: reset
        self.id_pair = []
        self.sim_score = []

        # stage 1: build voxel grid
        points = np.vstack([self.points[point_id].coord_3d for point_id in self.points])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
        voxel_to_point = defaultdict(list)
        
        # stage 2: assign points to voxels
        for point_id in self.points:
            point = self.points[point_id]
            voxel = voxel_grid.get_voxel(point.coord_3d)
            voxel = tuple(voxel.tolist())
            voxel_to_point[voxel].append(point)

        # stage 3: for each voxel, build point pair edges and calculate similarity
        for voxel in voxel_to_point:
            points_in_voxel = voxel_to_point[voxel] # a list of Point objects
            if len(points_in_voxel) <= 1:
                continue
            if len(points_in_voxel)%2 == 1:
                points_in_voxel = points_in_voxel[:-1]
            
            # randomly divide points into two groups
            random.shuffle(points_in_voxel)
            group1 = points_in_voxel[:len(points_in_voxel)//2]
            group2 = points_in_voxel[len(points_in_voxel)//2:]

            for pt1, pt2 in zip(group1, group2):
                # smaller id first
                if pt1.id > pt2.id: 
                    pt1, pt2 = pt2, pt1
                self.id_pair.append((pt1.id, pt2.id))
            
            self.sim_score += self.similarity(group1, group2)

        # stage 4: compress points with bipartite soft matching
        assert len(self.id_pair) == len(self.sim_score)
        id_pair_score = list(zip(self.id_pair, self.sim_score))
        id_pair_score.sort(key=lambda x: x[1], reverse=True)
        merge_count = int(len(id_pair_score) * r * 2) # since each pair has two points
        for (id1, id2), _ in id_pair_score[:merge_count]:
            # self.points[id1].update(self.points[id2])
            del self.points[id2]
    
    def pooling(self, voxel_size=0.1):
        
        # stage 0: reset
        self.id_pair = []
        self.sim_score = []

        # stage 1: build voxel grid
        points = np.vstack([self.points[point_id].coord_3d for point_id in self.points])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
        voxel_to_point = defaultdict(list)
        
        # stage 2: assign points to voxels
        for point_id in self.points:
            point = self.points[point_id]
            voxel = voxel_grid.get_voxel(point.coord_3d)
            voxel = tuple(voxel.tolist())
            voxel_to_point[voxel].append(point)

        # stage 3: for each voxel, build point pair edges and calculate similarity
        for voxel in voxel_to_point:
            points_in_voxel = voxel_to_point[voxel] # a list of Point objects
            avg_feat = torch.stack([pt.feature for pt in points_in_voxel]).mean(dim=0)
            if len(points_in_voxel) <= 1:
                continue
            for pt in points_in_voxel[1:]:
                del self.points[pt.id]
            first_id = points_in_voxel[0].id
            self.points[first_id].feature = avg_feat
    
    def return_features(self):
        features_dict = defaultdict(list)
        for point_id in self.points:
            point = self.points[point_id]
            frame_id = point.coord_2d[0]
            features_dict[frame_id].append(point.feature)
            
        features = []
        length = []
        
        for frame_id in range(self.max_frame+1):
            if len(features_dict[frame_id]) == 0: # to prevent empty frame, we use previous frame again.
                features_dict[frame_id] = features_dict[frame_id-1]
            features.append(torch.stack(features_dict[frame_id]))
            length.append(len(features_dict[frame_id]))
            
        return features, length
        
if __name__ == '__main__':
    
    scene = 'scene0784_00.pt'
    coords = torch.load(f'coord/{scene}')

    coords = [coord for i,coord in enumerate(coords) if i%50==0]

    print(len(coords))

    points = defaultdict(Point)
    point_id = 0

    for frame_id,frame_coords in enumerate(coords):
        for x,y in frame_coords:
            assert frame_coords[x,y].shape == (3,)
            points[point_id] = Point(point_id, (frame_id, x, y), frame_coords[x,y], torch.rand(1024))
            point_id += 1

    compressor = Compressor(points)
    print(compressor)
    compressor.compress(voxel_size=0.1, r=0.5)
    print(compressor)
    compressor.compress(voxel_size=0.15, r=0.5)
    print(compressor)
    import pdb; pdb.set_trace()